"""Hermes's own, automatic git must not execute a cloned repository's configuration.

The agent clones repositories onto its computer. Anything Hermes itself runs against such a repo
-- the session workspace snapshot, ``@diff`` references, the review pane, kanban worktrees,
``hermes -w`` and its cleanup -- must not trigger ``core.fsmonitor``, hooks, external diff / textconv
drivers, or clean/smudge filters defined in the repo's ``.git/config`` (upstream
NousResearch/hermes-agent#130661). Git the agent runs on purpose through its terminal tool is the
agent's own action and is deliberately not covered.

The tests build a real repo whose ``.git/config`` arms every sink to ``touch`` a marker file, and
assert no marker ever appears.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli._subprocess_compat import (
    FILTER_DISCOVERY_FAILED,
    NO_DRIVER_DIFF_FLAGS,
    harden_git_argv,
    noninteractive_git_env,
    noninteractive_repo_git_env,
)

pytestmark = pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not available",
)

_CLEAN_GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                  "GIT_CONFIG_NOSYSTEM": "1"}
_SINKS = ("fsmonitor", "hook", "extdiff", "textconv", "smudge", "clean", "ssh", "gpg")


def _git(repo: Path, *args: str) -> str:
    ident = ["-c", "user.email=a@b", "-c", "user.name=a"]
    return subprocess.run(["git", "-C", str(repo), *ident, *args], check=True, capture_output=True,
                          text=True, env=_CLEAN_GIT_ENV).stdout


def _fired(marker: Path) -> list[str]:
    return [sink for sink in _SINKS if Path(f"{marker}.{sink}").exists()]


def _evil_filter(marker: str, name: str = "evil") -> str:
    return (f'[filter "{name}"]\n\tsmudge = touch \'{marker}.smudge\'; cat\n'
            f'\tclean = touch \'{marker}.clean\'; cat\n\trequired = true\n')


def _make_repo(tmp: Path, attrs: str, extra_config: str = "", *, armed: bool = True) -> tuple[Path, str]:
    """Committed repo with ``.gitattributes`` = *attrs*; ``.git/config`` then gets every sink armed
    (when *armed*) plus *extra_config*. Returns (repo, marker stem)."""
    repo = tmp / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, env=_CLEAN_GIT_ENV)
    (repo / "README").write_text("hi\n")
    (repo / ".gitattributes").write_text(attrs)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    marker = (tmp / "MARKER").as_posix()
    if armed:
        hooks = repo / "evil-hooks"
        hooks.mkdir()
        # post-checkout fires on `worktree add`; reference-transaction on every ref update.
        for name in ("post-checkout", "reference-transaction"):
            hook = hooks / name
            hook.write_text(f"#!/bin/sh\ntouch '{marker}.hook'\n")
            hook.chmod(0o755)
        for key, value in {
            "core.fsmonitor": f"touch '{marker}.fsmonitor'",
            "core.hooksPath": str(hooks),
            "diff.evil.command": f"touch '{marker}.extdiff'",
            "diff.evil.textconv": f"touch '{marker}.textconv'; cat",
            "core.sshCommand": f"touch '{marker}.ssh'; false",
        }.items():
            _git(repo, "config", key, value)
        with open(repo / ".git" / "config", "a") as fh:
            fh.write(_evil_filter(marker))
    with open(repo / ".git" / "config", "a") as fh:
        fh.write(extra_config)
    return repo, marker


ATTRS = "* diff=evil filter=evil\n"


@pytest.fixture
def malicious(tmp_path):
    repo, marker = _make_repo(tmp_path, ATTRS)
    # Dirty working tree and a re-timestamped tracked file, so diff and status have work to do.
    (repo / "README").write_text("changed\n")
    return repo, Path(marker)


def test_control_raw_git_does_fire_the_sinks(malicious):
    """If this ever stops firing, the other tests prove nothing."""
    repo, marker = malicious
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "-C", str(repo), "status"], capture_output=True, env=env)
    subprocess.run(["git", "-C", str(repo), "diff"], capture_output=True, env=env)
    fired = set(_fired(marker))
    assert {"fsmonitor", "extdiff"} <= fired or {"fsmonitor", "textconv"} <= fired, fired


def test_env_pins_and_diff_flags():
    env = noninteractive_git_env({})
    values = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"]
              for i in range(int(env["GIT_CONFIG_COUNT"]))}
    assert values["core.fsmonitor"] == "false"
    assert values["core.hooksPath"] == os.devnull
    assert values["diff.external"] == ""
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert harden_git_argv(["diff", "--staged"])[:3] == ["diff", *NO_DRIVER_DIFF_FLAGS]
    assert harden_git_argv(["-C", "diff", "status"]) == ["-C", "diff", "status"]
    assert harden_git_argv(["status"]) == ["status"]


def test_env_replaces_inherited_config_injection():
    base = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.fsmonitor", "GIT_CONFIG_VALUE_0": "evil",
            "GIT_CONFIG_PARAMETERS": "'core.hooksPath=/tmp/x'"}
    env = noninteractive_git_env(base)
    assert "GIT_CONFIG_PARAMETERS" not in env
    assert "evil" not in [env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))]


def test_automatic_git_call_sites_never_run_repo_config(malicious, tmp_path, monkeypatch):
    import cli
    from agent.coding_context import build_coding_workspace_block
    from agent.context_references import _expand_git_reference
    from hermes_cli import kanban_db, web_git
    from tui_gateway import git_probe

    repo, marker = malicious
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)  # would mask core.sshCommand

    # Session workspace snapshot (status / log / rev-parse).
    block = build_coding_workspace_block(repo)
    assert "Root:" in block
    # `@diff`, `@staged`, `@git:1` references (diff drivers, clean filter, fsmonitor).
    for args in (["diff"], ["diff", "--staged"], ["log", "-1", "-p"]):
        err, _ = _expand_git_reference(SimpleNamespace(raw="@diff"), repo, args, "git")
        assert err is None, err
    # Review pane status / diff polling, and the TUI gateway's probes.
    code, _, _ = web_git._git(str(repo), ["status", "--porcelain=v2", "--branch", "-z"])
    assert code == 0
    web_git._git(str(repo), ["diff", "--", "README"])
    assert git_probe.run_git(str(repo), "branch", "--show-current") == "main"
    # Kanban worktree (checkout runs smudge + post-checkout).
    kanban_db._ensure_git_worktree(repo, tmp_path / "kwt", "kanban-safe")
    assert (tmp_path / "kwt" / "README").read_text() == "hi\n"
    assert _fired(marker) == []

    # `hermes -w`: create, probe, clean up, prune.
    info = cli._setup_worktree(str(repo), sync_base=False)
    assert info is not None and (Path(info["path"]) / "README").read_text() == "hi\n"
    os.utime(Path(info["path"]) / "README", (time.time() + 60, time.time() + 60))
    assert cli._worktree_is_dirty(info["path"]) is False  # the probe ran: a skipped one reads dirty
    cli._cleanup_worktree(info)
    _git(repo, "-c", "core.hooksPath=/dev/null", "branch", "pr-1")
    cli._prune_orphaned_branches(str(repo))
    assert "pr-1" not in _git(repo, "branch", "--format=%(refname:short)").split()
    assert _fired(marker) == []


@pytest.mark.parametrize("attrs, extra, refused", [
    pytest.param("README filter=evil\n", "", False, id="plain"),
    pytest.param("README filter=Evil\n", _evil_filter("{m}", "Evil"), False, id="case_collision"),
    pytest.param("README filter=evil\n", "[includeIf \"onbranch:safe\"]\n\tpath = evil.inc\n", True,
                 id="onbranch_include"),
    pytest.param("README filter=evil\n", "".join(f'[filter "f{i}"]\n\tclean = cat\n' for i in range(300)),
                 True, id="filter_flood"),
    pytest.param("README filter=evil\n", '[filter "evil"\n', True, id="broken_config"),
])
def test_repo_named_filters_never_run(tmp_path, attrs, extra, refused):
    """A filter driver is named by ``.gitattributes``, so fixed env pins cannot reach it. Subsection
    names are case-sensitive; discovery that cannot be trusted (any ``includeIf``, a huge filter
    inventory, a config git cannot parse) refuses the git call."""
    from hermes_cli import kanban_db
    marker = (tmp_path / "MARKER").as_posix()
    repo, _ = _make_repo(tmp_path, attrs, extra.replace("{m}", marker), armed=False)
    with open(repo / ".git" / "config", "a") as fh:
        fh.write(_evil_filter(marker))
    (repo / ".git" / "evil.inc").write_text(_evil_filter(marker))

    if refused:
        assert noninteractive_repo_git_env(repo) is None
        res = kanban_db._git_run(str(repo), ["worktree", "add", "-b", "x", str(tmp_path / "wt"), "HEAD"],
                                 timeout=30)
        assert (res.returncode, res.stderr) == (1, FILTER_DISCOVERY_FAILED)
        assert not (tmp_path / "wt").exists()
        assert _fired(Path(marker)) == []
        return

    from agent.coding_context import build_coding_workspace_block
    kanban_db._ensure_git_worktree(repo, tmp_path / "wt", "safe")
    assert (tmp_path / "wt" / "README").read_text() == "hi\n"
    (repo / "README").write_text("hi\n")  # same size, new mtime: status must re-hash it
    os.utime(repo / "README", (time.time() + 60, time.time() + 60))
    assert "Root:" in build_coding_workspace_block(repo)
    import cli
    assert cli._setup_worktree(str(repo), sync_base=False) is not None
    assert _fired(Path(marker)) == []


def test_refused_discovery_skips_status_in_workspace_block(tmp_path):
    from agent.coding_context import build_coding_workspace_block
    marker = (tmp_path / "MARKER").as_posix()
    repo, _ = _make_repo(tmp_path, "README filter=evil\n",
                         '[includeIf "onbranch:main"]\n\tpath = evil.inc\n', armed=False)
    (repo / ".git" / "evil.inc").write_text(_evil_filter(marker))
    (repo / "README").write_text("changed\n")
    assert "Root:" in build_coding_workspace_block(repo)
    assert _fired(Path(marker)) == []


def test_cli_worktree_setup_refuses_when_filters_undiscoverable(tmp_path):
    import cli
    marker = (tmp_path / "MARKER").as_posix()
    repo, _ = _make_repo(tmp_path, "README filter=evil\n", '[filter "evil"\n', armed=False)
    assert cli._setup_worktree(str(repo), sync_base=False) is None
    assert cli._worktree_is_dirty(str(repo)) is True  # fails safe toward "keep"
    assert _fired(Path(marker)) == []


# --- signing config and remote-chosen programs -------------------------------------------------

def _program(path: Path, marker_file: str, body: str = "") -> str:
    path.write_text(f"#!/bin/sh\ntouch '{marker_file}'\n{body}\nexit 1\n")
    path.chmod(0o755)
    return str(path)


def _signed_commit_repo(tmp: Path) -> tuple[Path, str]:
    """Repo whose HEAD carries a ``gpgsig`` header and whose config binds every gpg program (and the
    sign/verify switches) to a script that touches a marker."""
    repo, marker = _make_repo(tmp, "", armed=False)
    raw = _git(repo, "cat-file", "commit", "HEAD")
    head, _, msg = raw.partition("\n\n")
    sig = "gpgsig -----BEGIN PGP SIGNATURE-----\n \n fake\n -----END PGP SIGNATURE-----"
    forged = f"{head}\n{sig}\n\n{msg}"
    blob = tmp / "forged-commit"
    blob.write_text(forged)
    new = _git(repo, "hash-object", "-t", "commit", "-w", "--literally", str(blob)).strip()
    _git(repo, "update-ref", "refs/heads/main", new)
    script = _program(tmp / "gpg.sh", f"{marker}.gpg")
    for key in ("gpg.program", "gpg.openpgp.program", "gpg.x509.program", "gpg.ssh.program"):
        _git(repo, "config", key, script)
    _git(repo, "config", "log.showSignature", "true")
    _git(repo, "config", "commit.gpgSign", "true")
    _git(repo, "config", "tag.gpgSign", "true")
    return repo, marker


def test_control_raw_git_runs_gpg_program(tmp_path):
    repo, marker = _signed_commit_repo(tmp_path)
    subprocess.run(["git", "-C", str(repo), "log", "-1"], capture_output=True, env=_CLEAN_GIT_ENV)
    assert "gpg" in _fired(Path(marker))


def test_signing_config_never_runs_a_repo_program(tmp_path):
    from agent.coding_context import build_coding_workspace_block
    repo, marker = _signed_commit_repo(tmp_path)
    # Per-turn workspace probe (``git log -3``).
    assert "Root:" in build_coding_workspace_block(repo)
    assert _fired(Path(marker)) == []
    # ``harden_git_argv(['log', '-1'])`` under the repo-hardened env.
    env = noninteractive_repo_git_env(repo)
    assert env is not None
    subprocess.run(["git", "-C", str(repo), *harden_git_argv(["log", "-1"])], env=env,
                   capture_output=True, stdin=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(repo), *harden_git_argv(["show", "--show-signature", "-s"])],
                   env=env, capture_output=True, stdin=subprocess.DEVNULL)
    assert _fired(Path(marker)) == []
    # ``git commit`` with commit.gpgSign=true must not sign (and so not run the program).
    (repo / "README").write_text("again\n")
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=a", "-c", "user.email=a@b", "commit", "-aqm", "x"],
        env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stderr
    assert _fired(Path(marker)) == []


def test_env_pins_signing_keys():
    env = noninteractive_git_env({})
    values = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"]
              for i in range(int(env["GIT_CONFIG_COUNT"]))}
    for key in ("gpg.program", "gpg.openpgp.program", "gpg.x509.program", "gpg.ssh.program",
                "log.showSignature", "commit.gpgSign", "tag.gpgSign", "push.gpgSign"):
        assert values[key] == "false", key
    assert values["gpg.ssh.defaultKeyCommand"] == ""
