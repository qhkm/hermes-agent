"""Jentera: the agent's own skills cannot cover sign-ins, secrets or private forms.

Jentera keeps fixed rules for sign-ins, passwords, one-time codes, the private
form (``jentera-form``), the Vault and card details. A skill the agent writes
for itself is read as stronger than those rules, so on 1 Nov a learned skill
told the bot to type a username itself. Two parts:

- ``protected_topic`` / ``guard_skill_write``: ``skill_manage`` refuses a
  create, edit, patch or write_file whose text covers one of those topics,
  judged on the skill as it would read after the change (so a patch cannot
  finish a phrase the old text started). On only while
  ``JENTERA_SKILL_GUARD=on`` (the runner sets it from a settings switch);
  off by default, which changes nothing.
- ``guard_file_tool``: the general ``write_file`` and ``patch`` tools apply the
  same check to any file under a skills folder, and refuse writes to the
  folder's own records (``.usage.json``, ``.bundled_manifest``, ``.hub/``).
- ``sweep_protected_skills``: moves skills that cover the same topics out of
  the skills folder into ``~/aisar/removed-skills/<name>-<timestamp>/`` when
  they are agent-written (``created_by: agent``) or have no usage record at
  all and are neither shipped (bundled) nor hub-installed: those can only have
  been written by the agent, e.g. from the terminal. Never deletes. Logs counts only.
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
# Text is lower-cased and ``_-/.`` and whitespace become one space before matching.
# "log in"/"log on" count only as phrases about signing in ("log in to", "login"),
# never "a log on each run"; "username", "credentials" and "vault" alone are not
# enough (with a password, the password rule already refuses).
_TOPICS = (
    ("sign-in", r"\b(?:logins?|sign(?:ed|ing)? ?ins?|signins?|log ?into|"
                r"log(?:ged|ging)? ?(?:in|on) ?(?:to|into|with|as|using)|"
                r"log ?in (?:page|form|screen|details|button|link|flow|session)s?|"
                r"sign(?:ed|ing)? ?on ?to|single sign ?on|log masuk|daftar masuk)\b"),
    ("passwords", r"\b(?:passwords?|pass ?words?|passcodes?|passphrases?|kata laluan)\b"),
    ("one-time codes", r"\b(?:otps?|tacs?|2fa|mfa|two ?factor|one ?time (?:code|password|pin|passcode)s?|verification codes?|security codes?|auth(?:entication)? codes?|kod pengesahan)\b"),
    ("private forms", r"\b(?:jentera ?form|private forms?|borang peribadi)\b"),
    ("the Vault", r"\b(?:jentera ?records vault|jentera vault)\b"),
    ("card details", r"\b(?:cvvs?|cvcs?|cvv2|card numbers?|card details|credit cards?|debit cards?|card expiry|nombor kad)\b"),
)
_COMPILED = tuple((topic, re.compile(pattern)) for topic, pattern in _TOPICS)

REFUSAL = (
    "Not saved: Jentera has its own built-in rules for sign-ins, passwords, one-time codes, "
    "private forms (jentera-form), the Vault and card details, and a saved skill cannot cover them "
    "(this one covers {topic}). Follow Jentera's built-in rules instead: when a page needs something "
    "only the owner can type, ask with jentera-form. Save skills only for other kinds of work."
)


METADATA_REFUSAL = (
    "Not saved: the skills folder's own records (.usage.json, .bundled_manifest, .hub) "
    "are kept by Hermes and cannot be written with file tools."
)

def guard_enabled() -> bool:
    """On only for exactly ``on`` (any case)."""
    return os.environ.get(GUARD_ENV, "").strip().lower() == "on"


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


def _after_patch(name, file_path, old_string, new_string, replace_all) -> Optional[str]:
    """The skill file's text after ``skill_manage`` patch, or None when the patch cannot apply."""
    if not name or not old_string or new_string is None:
        return None
    from tools.fuzzy_match import fuzzy_find_and_replace
    from tools.skill_manager_tool import _find_skill, _resolve_skill_target

    found = _find_skill(name)
    if not found:
        return None
    skill_dir = Path(found["path"])
    if file_path:
        target, err = _resolve_skill_target(skill_dir, file_path)
        if err or target is None:
            return None
    else:
        target = skill_dir / "SKILL.md"
    try:
        current = target.read_text(encoding="utf-8")
    except OSError:
        return None
    new_content, _count, _strategy, error = fuzzy_find_and_replace(current, old_string, new_string, replace_all)
    return None if error else new_content

def guard_skill_write(
    action: str,
    name: Optional[str] = None,
    content: Optional[str] = None,
    new_string: Optional[str] = None,
    file_content: Optional[str] = None,
    file_path: Optional[str] = None,
    old_string: Optional[str] = None,
    replace_all: bool = False,
) -> Optional[str]:
    """The refusal message for a skill write that covers a protected topic, or None to let it through.

    A patch is judged on the file as it would read afterwards as well as on its new text,
    so splitting a phrase across the old text and the patch does not get through.
    """
    if not guard_enabled() or action not in GUARDED_ACTIONS:
        return None
    after = _after_patch(name, file_path, old_string, new_string, replace_all) if action == "patch" else None
    topic = protected_topic(name, content, new_string, file_content, file_path, after)
    if topic is None:
        return None
    logger.info("jentera skill guard: refused a skill %s", action)
    return REFUSAL.format(topic=topic)

def _skill_roots() -> list:
    roots = []
    from tools.skill_manager_tool import _skills_dir

    roots.append(_skills_dir())
    try:
        from agent.skill_utils import get_all_skills_dirs

        roots.extend(get_all_skills_dirs())
    except Exception:
        pass
    out = []
    for root in roots:
        real = Path(os.path.realpath(os.path.expanduser(str(root))))
        if real not in out:
            out.append(real)
    return out

def skill_relpath(path: str) -> Optional[str]:
    """``path`` relative to the skills folder it lies in, or None when it lies in none."""
    real = Path(os.path.realpath(os.path.expanduser(str(path))))
    for root in _skill_roots():
        try:
            return str(real.relative_to(root))
        except ValueError:
            continue
    return None

def guard_file_write(path: str, after_text: Optional[str], *extra: Optional[str]) -> Optional[str]:
    """The refusal for a file-tool write to ``path`` (absolute) leaving ``after_text``, or None."""
    if not guard_enabled():
        return None
    rel = skill_relpath(path)
    if rel is None:
        return None
    if any(part.startswith(".") for part in Path(rel).parts):
        logger.info("jentera skill guard: refused a file write to skills records")
        return METADATA_REFUSAL
    topic = protected_topic(rel, after_text, *extra)
    if topic is None:
        return None
    logger.info("jentera skill guard: refused a file write under skills")
    return REFUSAL.format(topic=topic)

def _read_local(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

class _DryRunOps:
    """File operations that read the disk and keep every change in memory (V4A dry run)."""

    def __init__(self, resolve):
        self._resolve = resolve
        self.files: Dict[str, Optional[str]] = {}

    def _read(self, path):
        absolute = self._resolve(path)
        if absolute in self.files:
            return self.files[absolute]
        return _read_local(absolute)

    def read_file_raw(self, path):
        from tools.file_operations import ReadResult

        content = self._read(path)
        if content is None:
            return ReadResult(error=f"File not found: {path}")
        return ReadResult(content=content)

    def write_file(self, path, content):
        from tools.file_operations import WriteResult

        self.files[self._resolve(path)] = content
        return WriteResult()

    def delete_file(self, path):
        from tools.file_operations import WriteResult

        self.files[self._resolve(path)] = None
        return WriteResult()

    def move_file(self, src, dst):
        from tools.file_operations import WriteResult

        content = self._read(src)
        self.files[self._resolve(src)] = None
        self.files[self._resolve(dst)] = content if content is not None else ""
        return WriteResult()

def guard_file_tool(
    mode: str,
    resolve,
    path: Optional[str] = None,
    content: Optional[str] = None,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    replace_all: bool = False,
    patch: Optional[str] = None,
) -> Optional[str]:
    """The refusal for a ``write_file`` / ``patch`` call that would leave a skill covering a protected topic.

    ``mode`` is ``write``, ``replace`` or ``patch`` (V4A); ``resolve`` maps a tool path to an absolute one.
    Each touched file under a skills folder is judged as it would read after the change.
    """
    if not guard_enabled():
        return None
    if mode == "write":
        return guard_file_write(resolve(path), content) if path else None
    if mode == "replace":
        if not path:
            return None
        target = resolve(path)
        if skill_relpath(target) is None:
            return None
        after = None
        current = _read_local(target)
        if current is not None and old_string is not None and new_string is not None:
            from tools.fuzzy_match import fuzzy_find_and_replace

            new_content, _count, _strategy, error = fuzzy_find_and_replace(
                current, old_string, new_string, replace_all)
            after = None if error else new_content
        return guard_file_write(target, after, new_string)
    if mode == "patch" and patch:
        from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

        operations, error = parse_v4a_patch(patch)
        targets = []
        for op in operations or []:
            for p in (getattr(op, "file_path", None), getattr(op, "new_path", None)):
                if p:
                    targets.append(resolve(p))
        if error or not operations:
            # Nothing will be applied; still refuse a patch whose text alone covers a topic in skills.
            touched = [t for t in targets if skill_relpath(t) is not None]
            return guard_file_write(touched[0], patch) if touched else None
        if not any(skill_relpath(t) is not None for t in targets):
            return None
        dry = _DryRunOps(resolve)
        result = apply_v4a_operations(operations, dry)
        if not getattr(result, "success", False):
            for t in targets:
                refusal = guard_file_write(t, patch)
                if refusal:
                    return refusal
            return None
        for absolute, text in dry.files.items():
            if text is None:
                continue
            refusal = guard_file_write(absolute, text)
            if refusal:
                return refusal
    return None

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

    Skills whose usage record says ``created_by: agent`` are considered, and so
    are skills with no usage record at all that are neither bundled (shipped)
    nor hub-installed: Hermes records every skill it ships or installs, so
    those can only have been written by the agent (from the terminal or file
    tools). Hub-installed, bundled and owner-recorded skills are never touched. Each
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
        record = usage.get(name)
        if record is not None and not skill_usage._is_curator_managed_record(record):
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
