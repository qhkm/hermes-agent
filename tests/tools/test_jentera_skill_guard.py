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
    ("Look it up in the Vault", "the Vault"),
    ("Save the card number and CVV", "card details"),
])
def test_protected_topics(text, topic):
    assert guard.protected_topic(text) == topic


@pytest.mark.parametrize("text", [
    RECIPE_SKILL,
    "Write invoices with the business name at the top and the total in bold.",
    "Post on Threads at 9 am with three hashtags.",
    "Summarise the week's bookings by day.",
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
