"""Jentera: the agent's own skills cannot cover sign-ins, secrets or private forms.

Jentera keeps fixed rules for sign-ins, passwords, one-time codes, the private
form (``jentera-form``), the Vault and card details. A skill the agent writes
for itself is read as stronger than those rules, so on 1 Nov a learned skill
told the bot to type a username itself. Two parts:

- ``protected_topic`` / ``guard_skill_write``: ``skill_manage`` refuses a
  create, edit, patch or write_file whose text covers one of those topics.
  On only while ``JENTERA_SKILL_GUARD=on`` (the runner sets it from a settings
  switch); off by default, which changes nothing.
- ``sweep_protected_skills``: moves agent-written skills (``created_by: agent``)
  that cover the same topics out of the skills folder into
  ``~/aisar/removed-skills/<name>-<timestamp>/``. Never deletes. Logs counts only.
  ``python -m tools.jentera_skill_guard sweep`` runs it for the current
  HERMES_HOME and prints the counts as JSON.

Detection is a small list of words and commands and errs on the side of refusing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

GUARD_ENV = "JENTERA_SKILL_GUARD"
REMOVED_DIR_ENV = "JENTERA_REMOVED_SKILLS_DIR"
GUARDED_ACTIONS = frozenset({"create", "edit", "patch", "write_file"})
# Text read per skill when sweeping; a skill is small, this only bounds a stray large file.
SWEEP_MAX_FILE_BYTES = 512 * 1024
SWEEP_MAX_FILES = 50

# (topic, pattern) on text that is lower-cased with _ - / . collapsed to spaces.
_TOPICS = (
    ("sign-in", r"\b(?:sign ?in|sign ?on|log ?in|log ?on|logins?|signins?|log masuk|daftar masuk|usernames?|user ?name|credentials?)\b"),
    ("passwords", r"\b(?:passwords?|pass ?words?|passcodes?|passphrases?|kata laluan)\b"),
    ("one-time codes", r"\b(?:otps?|tacs?|2fa|mfa|two ?factor|one ?time (?:code|password|pin|passcode)s?|verification codes?|security codes?|auth(?:entication)? codes?|kod pengesahan)\b"),
    ("private forms", r"\b(?:jentera ?form|private forms?|borang peribadi)\b"),
    ("the Vault", r"\b(?:vaults?|jentera ?records vault)\b"),
    ("card details", r"\b(?:cvvs?|cvcs?|cvv2|card numbers?|card details|credit cards?|debit cards?|card expiry|nombor kad)\b"),
)
_COMPILED = tuple((topic, re.compile(pattern)) for topic, pattern in _TOPICS)

REFUSAL = (
    "Not saved: Jentera has its own built-in rules for sign-ins, passwords, one-time codes, "
    "private forms (jentera-form), the Vault and card details, and a saved skill cannot cover them "
    "(this one covers {topic}). Follow Jentera's built-in rules instead: when a page needs something "
    "only the owner can type, ask with jentera-form. Save skills only for other kinds of work."
)


def guard_enabled() -> bool:
    return os.environ.get(GUARD_ENV, "").strip().lower() in {"on", "1", "true", "yes"}


def _normalise(text: str) -> str:
    return re.sub(r"[\s_\-/.]+", " ", text.lower())


def protected_topic(*texts: Optional[str]) -> Optional[str]:
    """The first protected topic any of ``texts`` covers, or None."""
    for text in texts:
        if not text:
            continue
        flat = _normalise(str(text))
        for topic, pattern in _COMPILED:
            if pattern.search(flat):
                return topic
    return None


def guard_skill_write(
    action: str,
    name: Optional[str] = None,
    content: Optional[str] = None,
    new_string: Optional[str] = None,
    file_content: Optional[str] = None,
    file_path: Optional[str] = None,
) -> Optional[str]:
    """The refusal message for a skill write that covers a protected topic, or None to let it through."""
    if not guard_enabled() or action not in GUARDED_ACTIONS:
        return None
    topic = protected_topic(name, content, new_string, file_content, file_path)
    if topic is None:
        return None
    logger.info("jentera skill guard: refused a skill %s", action)
    return REFUSAL.format(topic=topic)


def _skill_text(skill_dir: Path) -> Iterable[str]:
    count = 0
    for path in sorted(skill_dir.rglob("*")):
        if count >= SWEEP_MAX_FILES:
            break
        try:
            if not path.is_file() or path.is_symlink() or path.stat().st_size > SWEEP_MAX_FILE_BYTES:
                continue
            count += 1
            yield path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue


def _removed_root() -> Path:
    configured = os.environ.get(REMOVED_DIR_ENV, "").strip()
    return Path(configured) if configured else Path.home() / "aisar" / "removed-skills"


def sweep_protected_skills(
    skills_dir: Optional[Path] = None,
    dest_root: Optional[Path] = None,
    now: Optional[float] = None,
) -> Dict[str, int]:
    """Move agent-written skills covering protected topics out of ``skills_dir``.

    Only skills whose usage record says ``created_by: agent`` are considered;
    hub-installed, bundled and owner-written skills are never touched. Each
    one is moved whole (never deleted) to ``dest_root/<name>-<timestamp>/``.
    Returns counts: ``checked`` (agent-written skills read), ``moved``, ``failed``.
    """
    from agent.skill_utils import is_excluded_skill_path, is_external_skill_path
    from tools import skill_usage

    base = Path(skills_dir) if skills_dir is not None else skill_usage._skills_dir()
    dest = Path(dest_root) if dest_root is not None else _removed_root()
    counts = {"checked": 0, "moved": 0, "failed": 0}
    if not base.exists():
        return counts
    usage = skill_usage.load_usage()
    try:
        hub = skill_usage._read_hub_installed_names()
    except Exception:
        hub = set()
    try:
        bundled = skill_usage._read_bundled_manifest_names()
    except Exception:
        bundled = set()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now if now is not None else time.time()))
    moved_names = []
    for skill_md in sorted(base.rglob("SKILL.md")):
        try:
            if is_excluded_skill_path(skill_md) or is_external_skill_path(skill_md):
                continue
        except Exception:
            continue
        skill_dir = skill_md.parent
        if skill_dir == base or not skill_dir.exists():
            continue
        name = skill_usage._read_skill_name(skill_md, fallback=skill_dir.name)
        if name in hub or name in bundled:
            continue
        if not skill_usage._is_curator_managed_record(usage.get(name)):
            continue
        counts["checked"] += 1
        if protected_topic(name, *_skill_text(skill_dir)) is None:
            continue
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "skill"
        target = dest / f"{safe}-{stamp}"
        suffix = 1
        while target.exists():
            suffix += 1
            target = dest / f"{safe}-{stamp}-{suffix}"
        try:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(skill_dir), str(target))
        except Exception:
            counts["failed"] += 1
            continue
        counts["moved"] += 1
        moved_names.append(name)
    for name in moved_names:
        try:
            skill_usage.forget(name)
        except Exception:
            pass
    if moved_names:
        try:
            from agent.prompt_builder import clear_skills_system_prompt_cache

            clear_skills_system_prompt_cache(clear_snapshot=True)
        except Exception:
            pass
    logger.info(
        "jentera skill sweep: checked=%d moved=%d failed=%d",
        counts["checked"], counts["moved"], counts["failed"],
    )
    return counts


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["sweep"]:
        sys.stderr.write("usage: python -m tools.jentera_skill_guard sweep\n")
        return 2
    print(json.dumps(sweep_protected_skills()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
