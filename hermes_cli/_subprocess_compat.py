"""Windows subprocess compatibility helpers.

Hermes is developed on Linux / macOS and tested natively on Windows too.
Several common subprocess patterns break silently-or-loudly on Windows:

* ``["npm", "install", ...]`` — on Windows ``npm`` is ``npm.cmd``, a batch
  shim.  ``subprocess.Popen(["npm", ...])`` fails with WinError 193
  ("not a valid Win32 application") because CreateProcessW can't run a
  ``.cmd`` file without ``shell=True`` or PATHEXT resolution.

* ``start_new_session=True`` — on POSIX, this maps to ``os.setsid()`` and
  actually detaches the child.  On Windows it's silently ignored; the
  Windows equivalent is ``CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS``
  creationflags, which Python only applies when you pass them explicitly.

* Console-window flashes — every ``subprocess.Popen`` of a ``.exe`` on
  Windows spawns a cmd window briefly unless ``CREATE_NO_WINDOW`` is
  passed.  Cosmetic but jarring for background daemons.

This module centralizes the platform-branching logic so the rest of the
codebase doesn't sprinkle ``if sys.platform == "win32":`` everywhere.

**All helpers are no-ops on non-Windows** — calling them in Linux/macOS
code paths is safe by design.  That's the "do no damage on POSIX"
guarantee.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

__all__ = [
    "IS_WINDOWS",
    "resolve_node_command",
    "windows_detach_flags",
    "windows_detach_flags_without_breakaway",
    "windows_hide_flags",
    "windows_detach_popen_kwargs",
    "bounded_git_probe",
    "bounded_probe_run",
    "harden_git_argv",
    "noninteractive_git_env",
    "noninteractive_repo_git_env",
    "FILTER_DISCOVERY_FAILED",
    "NO_DRIVER_DIFF_FLAGS",
]


IS_WINDOWS = sys.platform == "win32"


# -----------------------------------------------------------------------------
# Node ecosystem launcher resolution
# -----------------------------------------------------------------------------


def resolve_node_command(name: str, argv: Sequence[str]) -> list[str]:
    """Resolve a Node-ecosystem command name to an absolute-path argv.

    On Windows, commands like ``npm``, ``npx``, ``yarn``, ``pnpm``,
    ``playwright``, ``prettier`` ship as ``.cmd`` files (batch shims).
    ``subprocess.Popen(["npm", "install"])`` fails with WinError 193
    because CreateProcessW doesn't execute batch files directly.

    ``shutil.which(name)`` *does* resolve ``.cmd`` via PATHEXT and returns
    the fully-qualified path — which CreateProcessW accepts because the
    extension tells Windows to route through ``cmd.exe /c``.

    On POSIX ``shutil.which`` also returns a fully-qualified path when
    found.  That's a small change from bare-name resolution (the OS does
    its own PATH search) but functionally identical and has the side
    benefit of making the argv reproducible in logs.

    Behavior when the command is not on PATH:
    - On Windows: return the bare name — caller can still try with
      ``shell=True`` as a last resort, OR the subsequent Popen will
      raise FileNotFoundError with a readable error we want to surface.
    - On POSIX: same.  Bare ``npm`` on a Linux box without npm installed
      fails the same way it did before this function existed.

    Args:
        name: The command name to resolve (``npm``, ``npx``, ``node`` …).
        argv: The remaining arguments.  Must NOT include ``name`` itself —
            this function builds the full argv list.

    Returns:
        A list suitable for passing to subprocess.Popen/run/call.
    """
    resolved = shutil.which(name)
    if resolved:
        return [resolved, *argv]
    return [name, *argv]


# -----------------------------------------------------------------------------
# Detached / hidden process creation
# -----------------------------------------------------------------------------


# Win32 CreationFlags — defined here rather than imported from subprocess
# because CREATE_NO_WINDOW and DETACHED_PROCESS aren't guaranteed to be
# present on stdlib subprocess on older Pythons or non-Windows builds.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS = 0x00000008
_CREATE_NO_WINDOW = 0x08000000
# Escape any Win32 job object the parent process belongs to. Without this,
# a detached child still inherits its parent's job object membership, and
# when that parent (Electron, Tauri, Windows Terminal, the Desktop GUI's
# bootstrap-installer) dies, the OS tears down the whole job — taking the
# "detached" child with it. Critical for the post-update gateway watcher:
# Electron spawns the Tauri updater inside its own job, the updater spawns
# the watcher subprocess; without BREAKAWAY the watcher dies the instant
# Electron exits, so the gateway never gets respawned after a `hermes
# update` triggered from the GUI. See fix/windows-gateway-reliability.
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def windows_detach_flags() -> int:
    """Return Win32 creationflags that detach a child from the parent
    console and process group.  0 on non-Windows.

    Pair with ``start_new_session=False`` (default) when calling
    subprocess.Popen — on POSIX use ``start_new_session=True`` instead,
    which maps to ``os.setsid()`` in the child.

    Rationale:
    - ``CREATE_NEW_PROCESS_GROUP`` — child has its own process group so
      Ctrl+C in the parent console doesn't propagate.
    - ``DETACHED_PROCESS`` — child has no console at all.  Necessary for
      background daemons (gateway watchers, update respawners) because
      without it, closing the console kills the child.
    - ``CREATE_NO_WINDOW`` — suppress the brief cmd flash that would
      otherwise appear when launching a console app.  Redundant with
      DETACHED_PROCESS but explicit for clarity.
    - ``CREATE_BREAKAWAY_FROM_JOB`` — escape any job object the parent is
      in.  Electron (Desktop app) and Tauri (bootstrap installer) wrap
      their children in job objects; without breakaway, those children
      die when the parent process exits even if they were spawned with
      DETACHED_PROCESS.  This was the missing flag that made the
      post-update gateway respawn watcher silently die alongside the
      Tauri updater after the Electron Desktop's update flow finished.

    If a process is in a job that disallows breakaway (rare —
    JOB_OBJECT_LIMIT_BREAKAWAY_OK isn't set), CreateProcess returns
    ERROR_ACCESS_DENIED.  Python surfaces that as ``PermissionError``
    on the ``subprocess.Popen`` call.  Callers in this codebase already
    wrap detached spawns in ``try/except OSError`` and fall back to a
    cmd.exe wrapper, so the breakaway-denied case degrades gracefully
    rather than crashing.
    """
    if not IS_WINDOWS:
        return 0
    return (
        _CREATE_NEW_PROCESS_GROUP
        | _DETACHED_PROCESS
        | _CREATE_NO_WINDOW
        | _CREATE_BREAKAWAY_FROM_JOB
    )


def windows_detach_flags_without_breakaway() -> int:
    """Same as :func:`windows_detach_flags` minus ``CREATE_BREAKAWAY_FROM_JOB``.

    The docstring on :func:`windows_detach_flags` notes that a process in
    a job which disallows breakaway (no ``JOB_OBJECT_LIMIT_BREAKAWAY_OK``)
    will see ``ERROR_ACCESS_DENIED`` from CreateProcess, surfacing as
    ``OSError`` (``PermissionError``) on the ``subprocess.Popen`` call.
    Callers that want to recover — by retrying without the breakaway
    bit — can pair the two helpers symbolically rather than coding the
    ``& ~0x01000000`` magic at every site:

    .. code-block:: python

        try:
            subprocess.Popen(argv, creationflags=windows_detach_flags(), …)
        except OSError:
            subprocess.Popen(
                argv,
                creationflags=windows_detach_flags_without_breakaway(),
                …,
            )

    See ``gateway_windows.py::_spawn_detached`` for the canonical
    implementation of this pattern.  Returns 0 on non-Windows.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS | _CREATE_NO_WINDOW


def windows_hide_flags() -> int:
    """Return Win32 creationflags that merely hide the child's console
    window without detaching the child.  0 on non-Windows.

    Use for short-lived console apps spawned as part of a larger
    operation (``taskkill``, ``where``, version probes) where we want no
    flash but also want to collect stdout/exit code synchronously.

    The key difference from :func:`windows_detach_flags`: NO
    ``DETACHED_PROCESS`` — the child still inherits stdio handles so
    ``capture_output=True`` works.  ``DETACHED_PROCESS`` would sever
    stdio and break stdout capture.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NO_WINDOW


def windows_detach_popen_kwargs() -> dict:
    """Return a dict of Popen kwargs that detach a child on Windows and
    fall back to the POSIX equivalent (``start_new_session=True``) on
    Linux/macOS.

    Usage pattern:

    .. code-block:: python

        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            **windows_detach_popen_kwargs(),
        )

    This replaces the unsafe-on-Windows pattern:

    .. code-block:: python

        subprocess.Popen(..., start_new_session=True)

    which silently fails to detach on Windows (the flag is accepted but
    has no effect — the child stays attached to the parent's console
    and dies when the console closes).
    """
    if IS_WINDOWS:
        return {"creationflags": windows_detach_flags()}
    return {"start_new_session": True}


# -----------------------------------------------------------------------------
# Bounded, fail-open git probing (Windows post-kill deadlock guard)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Hardened environment for git that Hermes runs on its own
# -----------------------------------------------------------------------------

# Flags that neutralize *attribute-scoped* diff drivers on any diff-rendering git command. A
# malicious repo can name a driver in ``.gitattributes`` (``* diff=evil``) and point it at an
# arbitrary program via ``[diff "evil"] command=/textconv=`` in ``.git/config``; because the
# attacker chooses the name, ``GIT_CONFIG_KEY`` overrides in ``noninteractive_git_env`` cannot
# enumerate it -- only these flags do. ``--no-ext-diff`` kills ``command=``; ``--no-textconv`` kills
# ``textconv=``; each alone leaves the other live. Repository-named clean/smudge/process filters
# are handled separately by ``noninteractive_repo_git_env`` at repo-scoped automatic call sites.
NO_DRIVER_DIFF_FLAGS = ("--no-ext-diff", "--no-textconv")

# A missing object in a partial clone would otherwise trigger a lazy fetch (network) from a probe.
NO_LAZY_FETCH_ENV = {"GIT_NO_LAZY_FETCH": "1"}

# Only these subcommands accept ``NO_DRIVER_DIFF_FLAGS`` -- ``status`` and friends reject them
# (``unknown option``), so the helper gates on this set rather than blanket-prepending.
_DIFF_RENDERING_SUBCOMMANDS = frozenset({"diff", "show", "log", "blame"})

# Options that consume the FOLLOWING token, so that value is never mistaken for the subcommand
# (``-C diff`` is a path; ``-c diff=x`` is a config pair).
_GIT_VALUE_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}


def harden_git_argv(args: Sequence[str]) -> list[str]:
    """Copy of subcommand-first git *args* (no leading ``"git"``) with :data:`NO_DRIVER_DIFF_FLAGS`
    inserted right after a diff-rendering subcommand; other subcommands are returned unchanged.

    Pair with :func:`noninteractive_git_env`: the env layer disables fsmonitor/hooks/pager/editor/
    credential sinks, this closes the one class (attacker-named attribute drivers) env cannot reach.
    """
    out = list(args)
    i = 0
    while i < len(out):
        tok = out[i]
        if tok in _GIT_VALUE_OPTS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if tok in _DIFF_RENDERING_SUBCOMMANDS:
            return out[: i + 1] + list(NO_DRIVER_DIFF_FLAGS) + out[i + 1 :]
        return out  # first non-option token is a non-diff subcommand
    return out


# GIT_CONFIG_KEY_n/VALUE_n overrides for internal git children: no credential/askpass prompts, no
# repo-configured fsmonitor/hooks/pager/editor/external-diff programs.
_GIT_CONFIG_INJECT_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
_GIT_CONFIG_OVERRIDES = {
    "credential.helper": "",
    "core.askPass": "",
    "core.fsmonitor": "false",
    "core.untrackedCache": "false",
    "core.hooksPath": os.devnull,
    "core.pager": "cat",
    "core.editor": "true",
    "sequence.editor": "true",
    "diff.external": "",
    # ssh bypasses stdin=DEVNULL/GIT_TERMINAL_PROMPT and opens /dev/tty directly; BatchMode makes
    # it fail instead of prompting. Config layer, so an explicit GIT_SSH_COMMAND still wins.
    "core.sshCommand": "ssh -o BatchMode=yes",
    # Signature verification / signing runs a program the repo config names. ``log.showSignature``
    # (or ``--show-signature`` in a log/show) verifies any commit with a ``gpgsig`` header through
    # ``gpg.program`` (and the openpgp/x509/ssh variants); ``commit.gpgSign`` etc. sign through it.
    # ``false`` is a program that exits non-zero without side effects; the sign/verify switches are
    # also forced off so the program is rarely even reached.
    "gpg.program": "false",
    "gpg.openpgp.program": "false",
    "gpg.x509.program": "false",
    "gpg.ssh.program": "false",
    "gpg.ssh.defaultKeyCommand": "",
    "log.showSignature": "false",
    "commit.gpgSign": "false",
    "tag.gpgSign": "false",
    "push.gpgSign": "false",
}


def _user_safe_directories(base_env: "Mapping[str, str]") -> list[str]:
    """The user's configured ``safe.directory`` values, in git's own effective order.

    ``safe.directory`` is honoured only from global/system config, which
    :func:`noninteractive_git_env` blanks; without replaying it every internal git call fails
    "detected dubious ownership" on a repo owned by another uid. Order and empty reset markers are
    policy, so every value is kept verbatim (``-z`` keeps whitespace inside one entry). Best-effort:
    any failure yields no entries.
    """
    env = dict(base_env)
    for key in list(env):
        if key == "GIT_CONFIG_PARAMETERS" or key.startswith(_GIT_CONFIG_INJECT_PREFIXES):
            env.pop(key, None)
    env.pop("GIT_CONFIG_COUNT", None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    values: list[str] = []
    for scope in ("--system", "--global"):
        try:
            proc = subprocess.run(
                ["git", "config", scope, "-z", "--get-all", "safe.directory"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=5, stdin=subprocess.DEVNULL, env=env, check=False,
                creationflags=windows_hide_flags(),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0:
            continue
        records = proc.stdout.split("\0")
        if records and records[-1] == "":
            records.pop()
        values.extend(records)
    return values


def noninteractive_git_env(base: "Mapping[str, str] | None" = None) -> dict[str, str]:
    """Environment for *internal* git invocations that must never prompt or run repo code.

    Copy of ``base`` (default ``os.environ``) with ``GIT_TERMINAL_PROMPT=0``, ``GCM_INTERACTIVE=Never``
    and isolated git config: inherited ``GIT_CONFIG_*`` injection, global/system config, pagers,
    editors, fsmonitor, external diff and hooks are all disabled, so a cloned repository's own
    ``.git/config`` cannot execute a program just because Hermes looked at it. Pair with
    ``stdin=subprocess.DEVNULL``. Internal plumbing only -- the agent's terminal tool is the
    agent's own action and is deliberately not routed through this.
    """
    env = dict(base if base is not None else os.environ)
    # Captured before the isolation below rewrites GIT_CONFIG_GLOBAL/SYSTEM to /dev/null.
    safe_directories = _user_safe_directories(base if base is not None else os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    for key in list(env):
        if key == "GIT_CONFIG_PARAMETERS" or key.startswith(_GIT_CONFIG_INJECT_PREFIXES):
            env.pop(key, None)
    env.pop("GIT_CONFIG_COUNT", None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    env["GIT_EDITOR"] = "true"
    # core.gitProxy: git honours the first entry it reads (the repo's), so a config override cannot
    # win; the environment variable is consulted before config, and empty means "no proxy".
    env["GIT_PROXY_COMMAND"] = ""
    overrides = list(_GIT_CONFIG_OVERRIDES.items())
    overrides.extend(("safe.directory", value) for value in safe_directories)
    env["GIT_CONFIG_COUNT"] = str(len(overrides))
    for idx, (key, value) in enumerate(overrides):
        env[f"GIT_CONFIG_KEY_{idx}"] = key
        env[f"GIT_CONFIG_VALUE_{idx}"] = value
    return env


_FILTER_COMMAND_KEY = re.compile(r"^filter\..+\.(?:clean|smudge|process)$", re.IGNORECASE)
# Any ``includeIf`` is evaluated against the CURRENT checkout: ``onbranch:`` against the current
# branch, ``gitdir:`` against the current git dir (``.git/worktrees/<name>`` during ``worktree add``),
# so the spawned git can load filters this discovery never saw; refuse rather than half-harden.
# Global/system config is already /dev/null, so only repo-local includes reach this.
_INCLUDE_IF_KEY = re.compile(r"^includeif\..*\.path$", re.IGNORECASE)
# Programs a repo's config can bind to a REMOTE it names. ``core.gitProxy`` (git:// transport) is
# overridden to empty. ``remote.<name>.uploadpack`` / ``receivepack`` (fetch / push) and
# ``remote.<name>.vcs`` (``git-remote-<vcs>`` helper) cannot be overridden from the command-line
# config layer -- git keeps the FIRST ``uploadpack`` it reads ("more than one uploadpack given,
# using the first"), which is the repo's, and ``vcs`` has no "off" value -- so the presence of any
# of them refuses the call, like ``includeIf``.
_REMOTE_PROGRAM_KEY = re.compile(r"^remote\..+\.(uploadpack|receivepack|vcs)$", re.IGNORECASE)
_GIT_PROXY_KEY = re.compile(r"^core\.gitproxy$", re.IGNORECASE)
# Each discovered key costs two env entries; a repo with tens of thousands of filters would make
# every spawn fail with E2BIG ("Argument list too long"), so refuse past a generous cap.
_MAX_FILTER_KEYS = 256
# Stand-in stderr for a git call refused because filter discovery could not be trusted.
FILTER_DISCOVERY_FAILED = "git filter discovery failed"


def noninteractive_repo_git_env(
    cwd: "str | os.PathLike[str]",
    base: "Mapping[str, str] | None" = None,
) -> "dict[str, str] | None":
    """Harden internal git for one repository, including named clean/smudge/process filters.

    The static environment can pin fixed config keys such as core.fsmonitor and
    core.hooksPath, but a repository chooses filter driver names through .gitattributes.
    Discover the effective filter command keys for this checkout and append empty command
    overrides plus required=false to the already-isolated config block. Discovery is
    bounded and fail-closed: if filter discovery cannot be trusted, callers skip the
    automatic git operation instead of running with only partial hardening.
    """
    env = noninteractive_git_env(base)
    # (Not bounded_git_probe: rc 1 = "no filters" is a verdict, not a failure.)
    proc = bounded_probe_run(
        [
            "git", "-C", str(cwd), "config", "--includes", "--name-only", "-z",
            "--get-regexp", r"^(filter\..*\.(clean|smudge|process)|includeif\..*\.path"
            r"|remote\..*\.(uploadpack|receivepack|vcs)|core\.gitproxy)$",
        ],
        timeout=2, env=env,
    )
    if proc is None or proc.returncode not in (0, 1):
        return None

    keys: list[str] = []
    required: list[str] = []
    program_overrides: list[tuple[str, str]] = []
    seen: set[str] = set()
    # Dedup on the exact name ``--name-only`` prints: git lowercases section and variable but keeps
    # the subsection's case, and ``[filter "Evil"]`` is a different driver from ``[filter "evil"]``.
    for raw in proc.stdout.split("\0"):
        key = raw.strip()
        if _INCLUDE_IF_KEY.fullmatch(key):
            return None
        if _REMOTE_PROGRAM_KEY.fullmatch(key):
            return None
        if key and key not in seen and _GIT_PROXY_KEY.fullmatch(key):
            seen.add(key)
            program_overrides.append((key, ""))
            continue
        if not key or key in seen or not _FILTER_COMMAND_KEY.fullmatch(key):
            continue
        seen.add(key)
        keys.append(key)
        if len(keys) > _MAX_FILTER_KEYS:
            return None
        required_key = key.rsplit(".", 1)[0] + ".required"
        if required_key not in seen:
            seen.add(required_key)
            required.append(required_key)

    start = int(env["GIT_CONFIG_COUNT"])  # always set by noninteractive_git_env
    overrides = ([(key, "") for key in keys] + [(key, "false") for key in required]
                 + program_overrides)
    for offset, (key, value) in enumerate(overrides):
        env[f"GIT_CONFIG_KEY_{start + offset}"] = key
        env[f"GIT_CONFIG_VALUE_{start + offset}"] = value
    env["GIT_CONFIG_COUNT"] = str(start + len(overrides))
    return env


def _kill_git_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort terminate *proc* and, on Windows, its descendants.

    ``proc.kill()`` alone only terminates the PATH-resolved ``git`` launcher; a
    suspended descendant ``git.exe`` can survive holding duplicates of the
    captured pipe handles, which keeps the pipes from reaching EOF and leaks two
    reader threads + the process per fired timeout. ``taskkill /T /F`` takes the
    whole tree down so the bounded drain that follows can actually reach EOF.

    All failures are swallowed — this is cleanup on an already-failing path, and
    the caller's contract is to fail open. ``kill()`` can raise (access denied,
    already reaped); an unhandled raise here would escape the caller's ``except``
    handler and break that contract. The ``taskkill`` spawn itself cannot
    re-enter the deadlock class it fixes: it captures no pipes (DEVNULL), so its
    own timeout cleanup has no reader threads to join.
    """
    try:
        proc.kill()
    except OSError:
        pass
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=2,
                check=False,
                creationflags=windows_hide_flags(),
            )
        except Exception:
            pass


def bounded_probe_run(
    argv: Sequence[str], *, timeout: float, env: "Mapping[str, str] | None" = None,
) -> "subprocess.CompletedProcess[str] | None":
    """Deadlock-safe ``subprocess.run(argv, capture_output=True, timeout=...)`` for fail-open probes.

    Returns a ``CompletedProcess`` when the child finished within *timeout* (any exit code), or
    ``None`` on spawn failure or timeout; see :func:`bounded_git_probe` for why not ``run``.
    """
    _popen_kwargs = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
    try:
        proc = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=None if env is None else dict(env),
            **_popen_kwargs,
        )
    except Exception:
        return None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except Exception:
        _kill_git_process_tree(proc)
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        return None
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout or "", stderr or "")


def bounded_git_probe(
    argv: Sequence[str], *, timeout: float, env: "Mapping[str, str] | None" = None,
) -> str:
    """Run a short, throwaway ``git`` probe and return stripped stdout, or ``""``
    on ANY failure (nonzero exit, timeout, spawn error, decode error).

    This is the shared, deadlock-safe replacement for
    ``subprocess.run(["git", ...], timeout=...)`` at fail-open probe call sites
    (``tui_gateway.git_probe.run_git``, ``agent.coding_context._git``).

    Why not ``subprocess.run``: on Windows, ``run()``'s post-timeout cleanup
    calls an *unbounded* ``communicate()`` after killing git. Killing the
    PATH-resolved launcher can leave a suspended descendant ``git.exe`` holding
    duplicates of the captured stdout/stderr handles, so the pipes never reach
    EOF and the reader-thread join blocks forever. On the Desktop agent-build
    path (``_start_agent_build → _session_info → branch() → run_git``) that turned
    an optional branch label into ``agent initialization timed out``
    (issues #68609 / #66037).

    The bounded flow: an explicit ``communicate(timeout)``, then on any failure a
    tree-kill (see :func:`_kill_git_process_tree`) plus a bounded 1s post-kill
    drain; if the pipes are still held after that, they're abandoned (the orphaned
    reader threads are daemonic and cost nothing).

    The normal-path spawn contract mirrors the previous ``run`` call byte-for-byte:
    PIPE/PIPE/DEVNULL, ``text`` with UTF-8 ``errors="replace"`` decoding, and the
    hidden-window ``creationflags`` on Windows only. The child runs under
    :func:`noninteractive_git_env` unless the caller supplies a hardened *env*.
    """
    result = bounded_probe_run(
        argv, timeout=timeout,
        env={**(env if env is not None else noninteractive_git_env()), **NO_LAZY_FETCH_ENV})
    if result is None or result.returncode != 0:
        return ""
    return (result.stdout or "").strip()
