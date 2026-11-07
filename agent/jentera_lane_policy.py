"""Trusted per-run confinement. File operations never enter terminal backends."""
import hashlib
import json
import os
import stat
import signal
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
import threading

TOOLS = ("web_search", "read_file", "write_file")
POLICY = {"version": 1, "id": "headless-v1", "tools": list(TOOLS)}
POLICY_HASH = hashlib.sha256(json.dumps(POLICY, separators=(",", ":")).encode()).hexdigest()
MAX_BYTES = 1024 * 1024
# Directory descriptors only anchor openat operations; listing access is not
# required. Linux O_PATH permits execute-only trusted ancestors while retaining
# O_DIRECTORY/O_NOFOLLOW checks. Other platforms retain their existing behavior.
_DIRECTORY_FLAGS = getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY | os.O_NOFOLLOW

@dataclass(frozen=True)
class LanePolicy:
    read_roots: tuple
    write_roots: tuple
    hash: str = POLICY_HASH
    processes: set = field(default_factory=set, compare=False, repr=False)
    process_lock: object = field(default_factory=threading.Lock, compare=False, repr=False)

    def execution_ended(self):
        with self.process_lock:
            for process in list(self.processes):
                if process.poll() is not None:
                    self.processes.discard(process)
            return not self.processes

    @classmethod
    def parse(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict) or any(value.get(k) != v for k, v in POLICY.items()) or value.get("hash") != POLICY_HASH:
            raise ValueError("unsupported lane policy")
        def roots(key):
            result = value.get(key)
            if not isinstance(result, list) or not 1 <= len(result) <= 2:
                raise ValueError("invalid policy roots")
            for root in result:
                if not isinstance(root, str) or not os.path.isabs(root) or os.path.normpath(root) != root:
                    raise ValueError("invalid policy root")
                # The trusted runner creates roots before admission. Refuse links,
                # including links in ancestors, before opening the directory.
                fd = cls._walk_directory(root)
                os.close(fd)
            return tuple(result)
        read, write = roots("read_roots"), roots("write_roots")
        if not set(write).issubset(read):
            raise ValueError("write roots must be readable")
        return cls(read, write)

    @staticmethod
    def _walk_directory(path):
        fd = os.open("/", _DIRECTORY_FLAGS)
        try:
            for component in path.split("/"):
                if not component:
                    continue
                new = os.open(component, _DIRECTORY_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = new
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _open(self, path, writing):
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ValueError("invalid path")
        roots = self.write_roots if writing else self.read_roots
        parts = path.split("/")
        if any(part == ".." or part.startswith(".") for part in parts if part):
            raise ValueError("hidden and parent paths are forbidden")
        root = roots[0]
        if os.path.isabs(path):
            matches = [r for r in roots if path.startswith(r + "/")]
            if not matches:
                raise ValueError("path outside task roots")
            root = max(matches, key=len)
            path = path[len(root) + 1:]
        components = path.split("/")
        if any(not x for x in components):
            raise ValueError("invalid path")
        fd = self._walk_directory(root)
        try:
            for component in components[:-1]:
                new = os.open(component, _DIRECTORY_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = new
            # Do not truncate until regular-file and hardlink checks pass.
            flags = os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_WRONLY | os.O_CREAT if writing else os.O_RDONLY)
            result = os.open(components[-1], flags, 0o600, dir_fd=fd)
            info = os.fstat(result)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                os.close(result)
                raise ValueError("only regular, unlinked task files are allowed")
            return result
        finally:
            os.close(fd)

    def invoke(self, name, args):
        if name not in TOOLS:
            return json.dumps({"error": "tool refused by headless-v1"})
        try:
            allowed = {"web_search": {"query", "limit"}, "read_file": {"path"}, "write_file": {"path", "content"}}[name]
            if not isinstance(args, dict) or set(args) - allowed:
                raise ValueError("unsupported tool arguments")
            if name == "write_file" and "content" not in args:
                raise ValueError("complete file content is required")
            if name == "web_search":
                query = args.get("query")
                if not isinstance(query, str) or len(query) > 4000:
                    raise ValueError("invalid search query")
                limit = min(10, max(1, int(args.get("limit", 5))))
                return json.dumps({"success": True, "data": {"web": restricted_search(query, limit, self)}})
            writing = name == "write_file"
            content = args.get("content", "")
            if writing and (not isinstance(content, str) or len(content.encode()) > MAX_BYTES):
                raise ValueError("file exceeds task limit")
            fd = self._open(args.get("path"), writing)
            with os.fdopen(fd, "wb" if writing else "rb") as stream:
                if writing:
                    stream.truncate(0)
                    stream.write(content.encode())
                    return json.dumps({"success": True, "path": args.get("path")})
                data = stream.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ValueError("file exceeds task limit")
                return json.dumps({"content": data.decode("utf-8", errors="replace")})
        except Exception:
            # Never expose provider credentials, filesystem paths or environment.
            return json.dumps({"error": "headless tool refused or unavailable"})


def apply_policy(agent, policy):
    if policy is None:
        return
    agent.jentera_lane_policy = policy
    # Restricted file schemas describe the handler actually used. Never offer
    # append/offset/limit knobs whose stock backend semantics are unavailable.
    agent.tools = [t for t in agent.tools if t.get("function", {}).get("name") == "web_search"]
    for name in ("read_file", "write_file"):
        properties = {"path": {"type": "string", "description": "File path inside this task folder. No hidden paths, parents or symlinks."}}
        if name == "write_file":
            properties["content"] = {"type": "string", "description": "Complete UTF-8 file content (at most 1 MiB); replaces the file."}
        agent.tools.append({"type": "function", "function": {"name": name,
            "description": "Read a complete task UTF-8 file, at most 1 MiB." if name == "read_file" else "Write or replace a task UTF-8 file, at most 1 MiB.",
            "parameters": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}})
    agent.valid_tool_names = {t["function"]["name"] for t in agent.tools}
    agent._memory_manager = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False


def track_tool_future(agent, future):
    """A submitted/running Future owns work even before its thread registers."""
    with agent._tool_worker_threads_lock:
        if not isinstance(getattr(agent, "_jentera_tool_futures", None), set):
            agent._jentera_tool_futures = set()
        agent._jentera_tool_futures.add(future)
    def finished(done):
        with agent._tool_worker_threads_lock:
            agent._jentera_tool_futures.discard(done)
    future.add_done_callback(finished)


def restricted_search(query, limit, owner=None):
    """Fixed DDGS worker, not a shell/code facility or provider fallback.

    DDGS's native transport can hold the GIL, so reuse its existing disposable
    worker. Pass no model/profile/runtime secrets or test hooks into the child.
    The call owns and reaps the process before releasing tool ownership.
    """
    root = Path(__file__).resolve().parents[1]
    worker = root / "plugins" / "web" / "ddgs" / "_search_worker.py"
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", "HOME": "/nonexistent",
           "HERMES_HOME": "/nonexistent/jentera-headless", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": str(root)}
    proc = subprocess.Popen([sys.executable, str(worker)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        cwd="/", env=env, start_new_session=True)
    if owner is not None:
        with owner.process_lock:
            owner.processes.add(proc)
    try:
        output, _ = proc.communicate(json.dumps({"query": query, "safe_limit": limit}), timeout=30)
        if proc.returncode != 0 or len(output.encode()) > MAX_BYTES:
            raise ValueError("search unavailable")
        response = json.loads(output)
        if response.get("ok") is not True or not isinstance(response.get("results"), list):
            raise ValueError("search unavailable")
        return [{"title": str(hit.get("title", ""))[:500], "url": str(hit.get("url", ""))[:2000],
                 "description": str(hit.get("description", ""))[:4000], "position": index + 1}
                for index, hit in enumerate(response["results"][:limit]) if isinstance(hit, dict)]
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate(timeout=2)
        proc.wait(timeout=2)
        if owner is not None and proc.poll() is not None:
            with owner.process_lock:
                owner.processes.discard(proc)
