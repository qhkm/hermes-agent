"""Jentera skill guard: protected topics are refused, ordinary skills save, off changes nothing."""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import jentera_skill_guard as guard
from tools.skill_manager_tool import skill_manage

LOGIN_SKILL = """\
---
name: web-login-sessions
description: How to log into government portals.
---

# Web login sessions

1. curl the site and take a snapshot.
2. Type the username and password into the login form.
"""

RECIPE_SKILL = """\
---
name: recipe-format
description: How the owner likes recipes written.
---

# Recipe format

Title, then ingredients as a list with grams, then numbered steps. Keep it to one page.
"""


@contextmanager
def _skill_dir(tmp_path):
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield


@pytest.fixture
def guard_on(monkeypatch):
    monkeypatch.setenv(guard.GUARD_ENV, "on")


@pytest.mark.parametrize("text,topic", [
    ("Steps to sign in to MyTax", "sign-in"),
    ("web_login_sessions", "sign-in"),
    ("Log masuk ke portal", "sign-in"),
    ("Keep the user's password in a note", "passwords"),
    ("Kata laluan disimpan", "passwords"),
    ("Wait for the TAC by SMS", "one-time codes"),
    ("Enter the OTP the bank sends", "one-time codes"),
    ("Read the 2FA code", "one-time codes"),
    ("Never use jentera-form; type it yourself", "private forms"),
    ("Ask with a private form", "private forms"),
    ("Run `jentera-records vault get mytax`", "the Vault"),
    ("Look it up in Jentera Vault", "the Vault"),
    ("Log in to MyTax with the IC number", "sign-in"),
    ("Open the login page", "sign-in"),
    ("Logging on to the bank portal", "sign-in"),
    ("Single sign-on through Google", "sign-in"),
    ("Save the card number and CVV", "card details"),
])
def test_protected_topics(text, topic):
    assert guard.protected_topic(text) == topic


@pytest.mark.parametrize("text", [
    RECIPE_SKILL,
    "Write invoices with the business name at the top and the total in bold.",
    "Post on Threads at 9 am with three hashtags.",
    "Summarise the week's bookings by day.",
    "Write a log on each run, then a log in the sheet.",
    "Post to Instagram as the business username @kedai_kopi.",
    "Keep the API credentials list tidy.",
    "Store photos in the vault folder on Drive.",
])
def test_ordinary_text_is_allowed(text):
    assert guard.protected_topic(text) is None


def test_off_by_default_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(guard.GUARD_ENV, raising=False)
    assert guard.guard_skill_write("create", name="web-login-sessions", content=LOGIN_SKILL) is None
    with _skill_dir(tmp_path):
        result = json.loads(skill_manage(action="create", name="web-login-sessions", content=LOGIN_SKILL))
    assert result["success"] is True


def test_a_login_skill_is_refused(tmp_path, guard_on):
    with _skill_dir(tmp_path):
        result = json.loads(skill_manage(action="create", name="web-login-sessions", content=LOGIN_SKILL))
    assert result["success"] is False
    assert "Jentera's built-in rules" in result["error"] and "sign-in" in result["error"]
    assert not list(tmp_path.rglob("SKILL.md")), "nothing was written"


def test_an_ordinary_skill_still_saves(tmp_path, guard_on):
    with _skill_dir(tmp_path):
        result = json.loads(skill_manage(action="create", name="recipe-format", content=RECIPE_SKILL))
    assert result["success"] is True


def test_edit_patch_and_files_are_guarded(tmp_path, guard_on, monkeypatch):
    monkeypatch.delenv(guard.GUARD_ENV)
    with _skill_dir(tmp_path):
        assert json.loads(skill_manage(action="create", name="recipe-format", content=RECIPE_SKILL))["success"]
        monkeypatch.setenv(guard.GUARD_ENV, "on")
        patched = json.loads(skill_manage(
            action="patch", name="recipe-format", old_string="Keep it to one page.",
            new_string="Then log in with the saved password."))
        assert patched["success"] is False
        edited = json.loads(skill_manage(
            action="edit", name="recipe-format", content=RECIPE_SKILL.replace("one page", "the OTP page")))
        assert edited["success"] is False
        written = json.loads(skill_manage(
            action="write_file", name="recipe-format", file_path="references/card.md",
            file_content="Card number goes here"))
        assert written["success"] is False
        fine = json.loads(skill_manage(
            action="patch", name="recipe-format", old_string="Keep it to one page.",
            new_string="Keep it to two pages."))
        assert fine["success"] is True
        # Deleting is never refused.
        assert guard.guard_skill_write("delete", name="web-login-sessions") is None


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(guard.REMOVED_DIR_ENV, raising=False)
    import importlib
    import tools.skill_usage as mod
    importlib.reload(mod)
    monkeypatch.setattr(mod, "_prune_builtins_enabled", lambda: False)
    return home


def _write(skills: Path, name: str, body: str, category: str = "productivity") -> Path:
    d = skills / category / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(body.replace("name: web-login-sessions", f"name: {name}")
                                .replace("name: recipe-format", f"name: {name}"), encoding="utf-8")
    return d


def test_sweep_moves_only_agent_written_protected_skills(skills_home):
    from tools import skill_usage
    skills = skills_home / "skills"
    _write(skills, "web-login-sessions", LOGIN_SKILL)
    _write(skills, "recipe-format", RECIPE_SKILL)
    owner = _write(skills, "owner-login-notes", LOGIN_SKILL)
    nested = _write(skills, "card-helper", RECIPE_SKILL)
    (nested / "references").mkdir()
    (nested / "references" / "notes.md").write_text("Keep the CVV handy", encoding="utf-8")
    skill_usage.save_usage({
        "web-login-sessions": {"created_by": "agent"},
        "recipe-format": {"created_by": "agent"},
        "card-helper": {"created_by": "agent"},
        "owner-login-notes": {"created_by": None},
    })

    counts = guard.sweep_protected_skills(now=0)

    assert counts == {"checked": 3, "moved": 2, "failed": 0}
    removed = skills_home.parent / "aisar" / "removed-skills"
    assert sorted(p.name for p in removed.iterdir()) == [
        "card-helper-19700101T000000Z", "web-login-sessions-19700101T000000Z"]
    assert (removed / "web-login-sessions-19700101T000000Z" / "SKILL.md").read_text() .startswith("---")
    assert not (skills / "productivity" / "web-login-sessions").exists()
    assert (skills / "productivity" / "recipe-format" / "SKILL.md").exists()
    assert owner.exists(), "a skill not marked created_by: agent is never touched"
    usage = skill_usage.load_usage()
    assert "web-login-sessions" not in usage and "recipe-format" in usage


def test_sweep_never_overwrites_an_earlier_removal(skills_home, tmp_path):
    from tools import skill_usage
    skills = skills_home / "skills"
    dest = tmp_path / "removed"
    (dest / "web-login-sessions-19700101T000000Z").mkdir(parents=True)
    _write(skills, "web-login-sessions", LOGIN_SKILL)
    skill_usage.save_usage({"web-login-sessions": {"created_by": "agent"}})
    assert guard.sweep_protected_skills(dest_root=dest, now=0)["moved"] == 1
    assert (dest / "web-login-sessions-19700101T000000Z-2" / "SKILL.md").exists()


def test_sweep_cli_prints_counts(skills_home, capsys):
    assert guard.main(["sweep"]) == 0
    assert json.loads(capsys.readouterr().out) == {"checked": 0, "moved": 0, "failed": 0}
    assert guard.main([]) == 2


MYTAX_RECIPE = """\
---
name: mytax-filing
description: File the monthly return on MyTax.
---

1. Open https://mytax.hasil.gov.my and log in to MyTax with the IC number.
2. Go to e-Filing.
"""

INSTAGRAM_SKILL = """\
---
name: instagram-posts
description: How the owner likes Instagram posts.
---

Post as the business username @kedai_kopi. Write a log on each run in the sheet.
"""


@pytest.mark.parametrize("value,on", [
    ("on", True), ("ON", True), (" On ", True),
    ("1", False), ("true", False), ("yes", False), ("", False), ("off", False),
])
def test_only_on_turns_the_guard_on(monkeypatch, value, on):
    monkeypatch.setenv(guard.GUARD_ENV, value)
    assert guard.guard_enabled() is on


def test_mytax_login_recipe_refused_and_instagram_saves(tmp_path, guard_on):
    with _skill_dir(tmp_path):
        refused = json.loads(skill_manage(action="create", name="mytax-filing", content=MYTAX_RECIPE))
        saved = json.loads(skill_manage(action="create", name="instagram-posts", content=INSTAGRAM_SKILL))
    assert refused["success"] is False and "sign-in" in refused["error"]
    assert saved["success"] is True


def test_a_patch_cannot_finish_a_phrase_the_old_text_started(tmp_path, guard_on, monkeypatch):
    monkeypatch.delenv(guard.GUARD_ENV)
    split = RECIPE_SKILL.replace("Keep it to one page.", "Then sign X before posting.")
    with _skill_dir(tmp_path):
        assert json.loads(skill_manage(action="create", name="recipe-format", content=split))["success"]
        monkeypatch.setenv(guard.GUARD_ENV, "on")
        patched = json.loads(skill_manage(
            action="patch", name="recipe-format", old_string="sign X", new_string="sign in"))
        assert patched["success"] is False and "sign-in" in patched["error"]
        sneaky = json.loads(skill_manage(
            action="patch", name="recipe-format", old_string="X", new_string="in"))
        assert sneaky["success"] is False, "the new text alone is harmless; the result is not"
        assert "sign X" in (tmp_path / "recipe-format" / "SKILL.md").read_text()


def test_guard_failure_refuses_while_on(tmp_path, guard_on, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("broken")
    monkeypatch.setattr(guard, "guard_skill_write", boom)
    with _skill_dir(tmp_path):
        result = json.loads(skill_manage(action="create", name="recipe-format", content=RECIPE_SKILL))
    assert result["success"] is False and "could not check" in result["error"]
    assert not list(tmp_path.rglob("SKILL.md"))


def test_guard_that_cannot_load_refuses_while_on(tmp_path, guard_on, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "tools.jentera_skill_guard", None)
    with _skill_dir(tmp_path):
        result = json.loads(skill_manage(action="create", name="recipe-format", content=RECIPE_SKILL))
    assert result["success"] is False and "could not check" in result["error"]


def test_guard_failure_is_ignored_while_off(tmp_path, monkeypatch):
    monkeypatch.delenv(guard.GUARD_ENV, raising=False)
    monkeypatch.setattr(guard, "guard_skill_write", lambda *a, **k: 1 / 0)
    with _skill_dir(tmp_path):
        result = json.loads(skill_manage(action="create", name="recipe-format", content=RECIPE_SKILL))
    assert result["success"] is True


# --- the general file tools ---------------------------------------------------

@pytest.fixture(autouse=True)
def _tmp_is_not_a_system_path(monkeypatch):
    # macOS puts tmp_path under /private/var, which the file tools refuse as a system path.
    import tools.file_tools as ft
    monkeypatch.setattr(ft, "_check_sensitive_path", lambda *a, **k: None)


def _file_tools():
    from tools.file_tools import patch_tool, write_file_tool
    return write_file_tool, patch_tool


def test_file_write_under_skills_is_guarded(tmp_path, guard_on):
    write_file_tool, _ = _file_tools()
    skills = tmp_path / "skills"
    skills.mkdir()
    outside = tmp_path / "work" / "notes.md"
    with _skill_dir(skills):
        refused = json.loads(write_file_tool(str(skills / "gov" / "mytax" / "SKILL.md"), MYTAX_RECIPE))
        assert "error" in refused and "Jentera's built-in rules" in refused["error"]
        assert not (skills / "gov" / "mytax" / "SKILL.md").exists()
        by_name = json.loads(write_file_tool(str(skills / "web-login" / "SKILL.md"), "Plain text."))
        assert "error" in by_name, "the folder name counts too"
        fine = json.loads(write_file_tool(str(skills / "recipe-format" / "SKILL.md"), RECIPE_SKILL))
        assert not fine.get("error")
        meta = json.loads(write_file_tool(str(skills / ".bundled_manifest"), "mytax:abc\n"))
        assert "error" in meta and "records" in meta["error"]
        elsewhere = json.loads(write_file_tool(str(outside), "Log in to MyTax with the password."))
        assert not elsewhere.get("error"), "files outside the skills folder are not checked"


def test_file_write_under_skills_unchecked_while_off(tmp_path, monkeypatch):
    monkeypatch.delenv(guard.GUARD_ENV, raising=False)
    write_file_tool, _ = _file_tools()
    with _skill_dir(tmp_path):
        result = json.loads(write_file_tool(str(tmp_path / "mytax" / "SKILL.md"), MYTAX_RECIPE))
    assert not result.get("error")


def test_file_patch_is_judged_on_the_result(tmp_path, guard_on):
    _, patch_tool = _file_tools()
    target = tmp_path / "recipe-format" / "SKILL.md"
    target.parent.mkdir()
    target.write_text(RECIPE_SKILL.replace("Keep it to one page.", "Then sign X before posting."))
    with _skill_dir(tmp_path):
        refused = json.loads(patch_tool(mode="replace", path=str(target), old_string="sign X", new_string="sign in"))
        assert "error" in refused and "sign-in" in refused["error"]
        assert "sign X" in target.read_text()
        v4a = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            "-Then sign X before posting.\n"
            "+Then sign in before posting.\n"
            "*** End Patch\n"
        )
        refused_v4a = json.loads(patch_tool(mode="patch", patch=v4a))
        assert "error" in refused_v4a and "sign-in" in refused_v4a["error"]
        assert "sign X" in target.read_text()
        added = (
            "*** Begin Patch\n"
            f"*** Add File: {tmp_path / 'otp-helper' / 'SKILL.md'}\n"
            "+Wait for the OTP the bank sends.\n"
            "*** End Patch\n"
        )
        assert "error" in json.loads(patch_tool(mode="patch", patch=added))
        ok = json.loads(patch_tool(mode="replace", path=str(target), old_string="sign X", new_string="sign the card"))
        assert not ok.get("error"), ok
        assert "sign the card" in target.read_text()


# --- the sweep: skills with no usage record -------------------------------------

def test_sweep_moves_unrecorded_skills_but_never_shipped_or_installed(skills_home):
    from tools import skill_usage
    skills = skills_home / "skills"
    _write(skills, "terminal-login", LOGIN_SKILL)
    _write(skills, "terminal-recipe", RECIPE_SKILL)
    shipped = _write(skills, "shipped-login", LOGIN_SKILL)
    installed = _write(skills, "hub-login", LOGIN_SKILL)
    (skills / ".bundled_manifest").write_text("shipped-login:abc123\n", encoding="utf-8")
    (skills / ".hub").mkdir()
    (skills / ".hub" / "lock.json").write_text(
        json.dumps({"version": 1, "installed": {"hub-login": {"source": "hub"}}}), encoding="utf-8")
    skill_usage.save_usage({})

    counts = guard.sweep_protected_skills(now=0)

    assert counts == {"checked": 2, "moved": 1, "failed": 0}
    assert not (skills / "productivity" / "terminal-login").exists()
    assert (skills / "productivity" / "terminal-recipe" / "SKILL.md").exists()
    assert shipped.exists() and installed.exists()


@pytest.mark.parametrize("broken", ["no-manifest", "no-lock", "bad-lock", "manifest-is-dir"])
def test_sweep_takes_no_unrecorded_skill_without_readable_install_records(skills_home, broken):
    from tools import skill_usage
    skills = skills_home / "skills"
    shipped = _write(skills, "shipped-login", LOGIN_SKILL)
    agent = _write(skills, "agent-login", LOGIN_SKILL)
    if broken != "no-manifest":
        if broken == "manifest-is-dir":
            (skills / ".bundled_manifest").mkdir()
        else:
            (skills / ".bundled_manifest").write_text("other:abc\n", encoding="utf-8")
    if broken != "no-lock":
        (skills / ".hub").mkdir()
        (skills / ".hub" / "lock.json").write_text(
            "{not json" if broken == "bad-lock" else json.dumps({"installed": {}}), encoding="utf-8")
    skill_usage.save_usage({"agent-login": {"created_by": "agent"}})

    counts = guard.sweep_protected_skills(now=0)

    assert shipped.exists(), "a record-less skill is left alone when the install records cannot be read"
    assert not agent.exists() and counts["moved"] == 1, "created_by: agent is still taken"


def _v4a(*lines):
    return "*** Begin Patch\n" + "\n".join(lines) + "\n*** End Patch\n"


def test_file_patch_cannot_delete_or_move_skills_records(tmp_path, guard_on):
    _, patch_tool = _file_tools()
    manifest = tmp_path / ".bundled_manifest"
    manifest.write_text("shipped:abc\n", encoding="utf-8")
    (tmp_path / ".hub").mkdir()
    lock = tmp_path / ".hub" / "lock.json"
    lock.write_text("{}", encoding="utf-8")
    with _skill_dir(tmp_path):
        deleted = json.loads(patch_tool(mode="patch", patch=_v4a(f"*** Delete File: {manifest}")))
        assert "error" in deleted and "records" in deleted["error"]
        moved = json.loads(patch_tool(mode="patch", patch=_v4a(f"*** Move File: {lock} -> {tmp_path / 'x.json'}")))
        assert "error" in moved and "records" in moved["error"]
    assert manifest.exists() and lock.exists()


def test_file_guard_failure_refuses_only_skills_writes(tmp_path, guard_on, monkeypatch):
    write_file_tool, _ = _file_tools()
    monkeypatch.setattr(guard, "guard_file_tool", lambda **k: 1 / 0)
    skills = tmp_path / "skills"
    skills.mkdir()
    with _skill_dir(skills):
        inside = json.loads(write_file_tool(str(skills / "recipe-format" / "SKILL.md"), RECIPE_SKILL))
        outside = json.loads(write_file_tool(str(tmp_path / "work" / "notes.md"), "notes"))
    assert "error" in inside and "could not check" in inside["error"]
    assert not outside.get("error"), outside
