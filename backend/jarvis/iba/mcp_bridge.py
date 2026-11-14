"""External MCP bridge (REA 6.8.0), Agent Skills discovery, binary/MIPS probe, and Ghidra headless runner."""
import atexit
from collections import deque
import json
import os
from pathlib import Path
import queue
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SKILLS_DIRS = [
    Path.home() / ".agents" / "skills",
    Path.home() / ".claude" / "skills",
]
DEFAULT_MCP_CONFIGS = [
    Path.home() / "AppData" / "Roaming" / "devin" / "mcp_config.json",
    Path.home() / ".gemini" / "antigravity" / "mcp_config.json",
]
DEFAULT_GHIDRA_DIR = Path(r"D:\MGS3 Tools From Wasp\_Tools\ghidra_12.1.4_PUBLIC")
DEFAULT_JAVA21_DIR = Path(r"C:\Program Files\Eclipse Adoptium\jdk-21.0.13.9-hotspot")
DEFAULT_REA_SCRIPT = Path.home() / "AppData" / "Roaming" / "npm" / "node_modules" / "rea-agents" / "scripts" / "rea.mjs"


def _find_java21_home() -> Optional[str]:
    if DEFAULT_JAVA21_DIR.is_dir():
        return str(DEFAULT_JAVA21_DIR)
    adoptium = Path(r"C:\Program Files\Eclipse Adoptium")
    if adoptium.is_dir():
        for child in sorted(adoptium.iterdir(), reverse=True):
            if child.is_dir() and "jdk-21" in child.name.lower():
                return str(child)
    return os.environ.get("JAVA_HOME")


def load_rea_server_config() -> Dict[str, Any]:
    """Resolve command, args, and environment for the REA MCP server."""
    for cfg_path in DEFAULT_MCP_CONFIGS:
        if cfg_path.is_file():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
                servers = data.get("mcpServers") or data.get("servers") or {}
                if isinstance(servers, dict) and "rea" in servers:
                    entry = dict(servers["rea"])
                    env = dict(entry.get("env") or {})
                    if not env.get("JAVA_HOME") or not Path(env["JAVA_HOME"]).is_dir():
                        j21 = _find_java21_home()
                        if j21:
                            env["JAVA_HOME"] = j21
                    if not env.get("GHIDRA_INSTALL_DIR") and DEFAULT_GHIDRA_DIR.is_dir():
                        env["GHIDRA_INSTALL_DIR"] = str(DEFAULT_GHIDRA_DIR)
                    entry["env"] = env
                    return entry
            except Exception:
                pass

    node = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
    env: Dict[str, str] = {}
    if DEFAULT_GHIDRA_DIR.is_dir():
        env["GHIDRA_INSTALL_DIR"] = str(DEFAULT_GHIDRA_DIR)
    j21 = _find_java21_home()
    if j21:
        env["JAVA_HOME"] = j21
    return {
        "command": node,
        "args": [str(DEFAULT_REA_SCRIPT), "mcp"],
        "env": env,
    }


def _parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, int):
        return value
    s = str(value).strip()
    return int(s, 0)


def _compact_summary(text: str, limit: int = 160) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


class MCPBridge:
    """Persistent stdio JSON-RPC client for external MCP servers (currently REA)."""

    def __init__(self, server_config: Optional[Dict[str, Any]] = None):
        self.server_config = server_config
        self._startup_lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_lines: deque = deque(maxlen=80)
        self._pending: Dict[int, Tuple[threading.Event, Dict[str, Any]]] = {}
        self._next_id = 1
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        atexit.register(self.close)

    def _ensure_started(self, timeout: float = 25.0) -> None:
        with self._startup_lock:
            if self._initialized and self._proc and self._proc.poll() is None:
                return
            self._close_locked()
            cfg = self.server_config or load_rea_server_config()
            cmd = cfg.get("command")
            args = list(cfg.get("args") or [])
            if not cmd or not Path(cmd).exists():
                raise RuntimeError(f"MCP command not found: {cmd!r}")
            if args and args[0].endswith(".mjs") and not Path(args[0]).exists():
                raise RuntimeError(f"REA MCP script not found: {args[0]!r}")

            env = dict(os.environ)
            for k, v in (cfg.get("env") or {}).items():
                if v is not None:
                    env[str(k)] = str(v)

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._proc = subprocess.Popen(
                [cmd, *args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
            self._reader_thread = threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True)
            self._stderr_thread = threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True)
            self._reader_thread.start()
            self._stderr_thread.start()

            self._rpc(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "jarvis-iba", "version": "1.0.0"},
                },
                timeout=timeout,
            )
            self._notify("notifications/initialized", {})
            listed = self._rpc("tools/list", {}, timeout=timeout)
            tools = listed.get("tools") if isinstance(listed, dict) else []
            self._tools = {t["name"]: t for t in tools if isinstance(t, dict) and "name" in t}
            self._initialized = True

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        if not proc or not proc.stdout:
            return
        try:
            while True:
                line = proc.stdout.readline()
                if line == "":
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict) or "id" not in msg:
                    continue
                req_id = msg["id"]
                with self._io_lock:
                    slot = self._pending.get(req_id)
                if slot is not None:
                    ev, holder = slot
                    holder["msg"] = msg
                    ev.set()
        except Exception:
            pass

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        if not proc or not proc.stderr:
            return
        try:
            while True:
                line = proc.stderr.readline()
                if line == "":
                    break
                line = line.rstrip()
                if line:
                    self._stderr_lines.append(line)
        except Exception:
            pass

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        with self._io_lock:
            proc = self._proc
            if not proc or not proc.stdin:
                raise RuntimeError("MCP process stdin is closed")
            payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()

    def _rpc(self, method: str, params: Dict[str, Any], timeout: float = 60.0) -> Any:
        with self._io_lock:
            proc = self._proc
            if not proc or proc.poll() is not None or not proc.stdin:
                raise RuntimeError("MCP process is not running")
            req_id = self._next_id
            self._next_id += 1
            ev = threading.Event()
            holder: Dict[str, Any] = {}
            self._pending[req_id] = (ev, holder)
            try:
                payload = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
                proc.stdin.write(payload + "\n")
                proc.stdin.flush()
            except Exception:
                self._pending.pop(req_id, None)
                raise

        signaled = ev.wait(timeout=timeout)
        with self._io_lock:
            self._pending.pop(req_id, None)
        if not signaled:
            tail = " | ".join(list(self._stderr_lines)[-5:])
            raise TimeoutError(f"MCP call {method!r} timed out after {timeout}s" + (f" ({tail})" if tail else ""))
        msg = holder.get("msg") or {}
        if "error" in msg and msg["error"] is not None:
            err = msg["error"]
            raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")
        return msg.get("result")

    def close(self) -> None:
        with self._startup_lock:
            self._close_locked()

    def _close_locked(self) -> None:
        self._initialized = False
        with self._io_lock:
            for req_id, (ev, holder) in list(self._pending.items()):
                holder["msg"] = {"error": {"code": -1, "message": "MCP bridge closed"}}
                ev.set()
            self._pending.clear()
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    def list_tools(self, refresh: bool = False) -> List[Dict[str, Any]]:
        self._ensure_started()
        if refresh:
            listed = self._rpc("tools/list", {}, timeout=30.0)
            tools = listed.get("tools") if isinstance(listed, dict) else []
            with self._lock:
                self._tools = {t["name"]: t for t in tools if isinstance(t, dict) and "name" in t}
        return list(self._tools.values())

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        if not name:
            return None
        if name in self._tools:
            return self._tools[name]
        try:
            self._ensure_started()
        except Exception:
            return None
        return self._tools.get(name)

    def has_tool(self, name: str, connect: bool = True) -> bool:
        if not name:
            return False
        if name in self._tools:
            return True
        if not connect:
            return False
        return self.get_tool_schema(name) is not None

    def validate_arguments(self, tool_name: str, arguments: Any) -> Tuple[bool, str]:
        if not isinstance(arguments, dict):
            return False, f"Tool {tool_name!r} arguments must be a JSON object"
        schema = self.get_tool_schema(tool_name)
        if not schema:
            return False, f"Unknown MCP tool {tool_name!r}; run discover_tools first"
        inp = schema.get("inputSchema") or {}
        props = inp.get("properties") or {}
        required = list(inp.get("required") or [])
        missing = [k for k in required if k not in arguments or arguments[k] is None]
        if missing:
            return False, f"Missing required arguments for {tool_name!r}: {', '.join(missing)}. Use inspect_tool({tool_name!r}) to see schema."
        if inp.get("additionalProperties") is False and props:
            extra = sorted(set(arguments) - set(props))
            if extra:
                return False, f"Unknown arguments for {tool_name!r}: {', '.join(extra)}. Use inspect_tool({tool_name!r}) to see schema."
        return True, ""

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None, timeout: float = 90.0) -> Dict[str, Any]:
        args = dict(arguments or {})
        self._ensure_started()
        ok, err = self.validate_arguments(tool_name, args)
        if not ok:
            return {"ok": False, "tool": tool_name, "error": err}

        res = self._rpc("tools/call", {"name": tool_name, "arguments": args}, timeout=timeout)
        is_error = bool(res.get("isError")) if isinstance(res, dict) else False
        content = res.get("content") if isinstance(res, dict) else []
        texts: List[str] = []
        parsed_blocks: List[Any] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = str(block.get("text") or "")
                    texts.append(txt)
                    try:
                        parsed_blocks.append(json.loads(txt))
                    except ValueError:
                        pass
        payload: Any
        if len(parsed_blocks) == 1:
            payload = parsed_blocks[0]
        elif parsed_blocks:
            payload = parsed_blocks
        else:
            payload = "\n".join(texts)

        result: Dict[str, Any] = {
            "ok": not is_error,
            "tool": tool_name,
            "result": payload,
        }
        if isinstance(payload, dict):
            for k in ("evidence_id", "binary_id", "session_id", "status", "tool"):
                if k in payload and k not in result:
                    result[k] = payload[k]
            if is_error and "error" in payload:
                result["error"] = payload["error"]
        elif is_error:
            result["error"] = str(payload)[:800]
        return result


_BRIDGE_SINGLETON: Optional[MCPBridge] = None
_BRIDGE_LOCK = threading.Lock()


def get_mcp_bridge() -> MCPBridge:
    global _BRIDGE_SINGLETON
    with _BRIDGE_LOCK:
        if _BRIDGE_SINGLETON is None:
            _BRIDGE_SINGLETON = MCPBridge()
        return _BRIDGE_SINGLETON


def _parse_frontmatter(md_text: str) -> Tuple[Dict[str, str], str]:
    if not md_text.startswith("---"):
        return {}, md_text
    end = md_text.find("\n---", 3)
    if end == -1:
        return {}, md_text
    meta: Dict[str, str] = {}
    for line in md_text[3:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, md_text[end + 4 :].lstrip()


def _skill_dirs(workspace: Optional[Path] = None) -> List[Path]:
    dirs: List[Path] = []
    if workspace:
        for candidate in (workspace / ".agents" / "skills", workspace / ".claude" / "skills"):
            if candidate.is_dir():
                dirs.append(candidate)
    for d in DEFAULT_SKILLS_DIRS:
        if d.is_dir() and d not in dirs:
            dirs.append(d)
    return dirs


def _scan_skills(query: str = "", workspace: Optional[Path] = None) -> List[Dict[str, Any]]:
    q_tokens = [t for t in query.lower().split() if t]
    items: List[Dict[str, Any]] = []
    seen = set()
    for base in _skill_dirs(workspace):
        for child in sorted(base.iterdir()):
            skill_md = child / "SKILL.md"
            if not child.is_dir() or not skill_md.is_file():
                continue
            meta, body = _parse_frontmatter(skill_md.read_text(encoding="utf-8", errors="replace"))
            name = meta.get("name") or child.name
            if name in seen:
                continue
            desc = meta.get("description") or _compact_summary(body, 180)
            hay = f"{name} {desc}".lower()
            score = sum(3 if t in name.lower() else 1 for t in q_tokens if t in hay) if q_tokens else 1
            if q_tokens and score == 0:
                continue
            refs_dir = child / "references"
            refs = sorted(p.name for p in refs_dir.glob("*.md")) if refs_dir.is_dir() else []
            seen.add(name)
            items.append({
                "name": name,
                "source": "skill",
                "summary": _compact_summary(desc, 160),
                "path": str(skill_md),
                "references": refs[:12],
                "_score": score,
            })
    items.sort(key=lambda x: (-x.pop("_score", 0), x["name"]))
    return items


def _python_doc_summary(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception:
        return ""
    m = re.search(r'(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', text, flags=re.S)
    if m:
        return _compact_summary(m.group(1), 160)
    comments = []
    for line in text.splitlines()[:10]:
        s = line.strip()
        if s.startswith("#"):
            comments.append(s.lstrip("#").strip())
    return _compact_summary(" ".join(comments), 160)


def _scan_workspace_tools(workspace: Optional[Path], query: str = "") -> List[Dict[str, Any]]:
    if not workspace or not workspace.is_dir():
        return []
    q_tokens = [t for t in query.lower().split() if t]
    items: List[Dict[str, Any]] = []
    candidates: List[Path] = []
    for sub in ("tools", "waspo", "scripts", "bin"):
        d = workspace / sub
        if d.is_dir():
            candidates.extend(sorted(d.iterdir())[:40])
    for p in sorted(workspace.iterdir())[:40]:
        if p.is_file() and p.suffix.lower() in (".py", ".ps1", ".bat"):
            candidates.append(p)
    seen = set()
    for p in candidates:
        if not p.is_file() or p.name.startswith("__"):
            continue
        if p.suffix.lower() not in (".py", ".ps1", ".bat", ".exe"):
            continue
        try:
            rel = str(p.relative_to(workspace)).replace("\\", "/")
        except ValueError:
            rel = p.name
        if rel in seen:
            continue
        seen.add(rel)
        summary = _python_doc_summary(p) if p.suffix.lower() == ".py" else f"Workspace executable/script ({p.suffix.lower()})"
        hay = f"{rel} {summary}".lower()
        score = sum(3 if t in rel.lower() else 1 for t in q_tokens if t in hay) if q_tokens else 1
        if q_tokens and score == 0:
            continue
        items.append({
            "name": rel,
            "source": "workspace",
            "summary": summary or f"Workspace tool {rel}",
            "path": str(p),
            "_score": score,
        })
    items.sort(key=lambda x: (-x.pop("_score", 0), x["name"]))
    return items


def discover_tools(
    query: str = "",
    source: str = "all",
    workspace: Optional[Path] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Search REA MCP tools, ~/.agents/skills, and workspace scripts with compact summaries."""
    src = (source or "all").strip().lower()
    if src not in {"all", "rea", "skills", "workspace"}:
        raise ValueError("source must be one of: all, rea, skills, workspace")
    cap = max(1, min(50, int(limit or 20)))
    q_tokens = [t for t in (query or "").lower().split() if t]

    results: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    errors: List[str] = []

    if src in {"all", "rea"}:
        try:
            rea_tools = get_mcp_bridge().list_tools()
            counts["rea_total"] = len(rea_tools)
            scored = []
            for t in rea_tools:
                name = t.get("name", "")
                desc = t.get("description", "")
                inp = t.get("inputSchema") or {}
                props = list((inp.get("properties") or {}).keys())
                req = list(inp.get("required") or [])
                hay = f"{name} {desc}".lower()
                if q_tokens:
                    score = sum(4 if t_tok in name.lower() else (1 if t_tok in hay else 0) for t_tok in q_tokens)
                    if score == 0:
                        continue
                else:
                    score = 2 if name in {"open_binary", "binary_overview", "analyze_function", "inspect_pe_imports", "inspect_analysis_view", "close_binary"} else 1
                scored.append((
                    -score,
                    name,
                    {
                        "name": name,
                        "source": "rea",
                        "summary": _compact_summary(desc, 150),
                        "required_args": req,
                        "args": props[:10],
                    },
                ))
            scored.sort(key=lambda row: (row[0], row[1]))
            counts["rea_matched"] = len(scored)
            results.extend(row[2] for row in scored)
        except Exception as exc:
            errors.append(f"REA MCP unavailable: {exc}")

    if src in {"all", "skills"}:
        skill_hits = _scan_skills(query, workspace)
        counts["skills_matched"] = len(skill_hits)
        results.extend(skill_hits)

    if src in {"all", "workspace"}:
        ws_hits = _scan_workspace_tools(workspace, query)
        counts["workspace_matched"] = len(ws_hits)
        results.extend(ws_hits)

    # Blend sensibly when source=='all' and query is empty so skills/workspace are visible alongside REA
    if src == "all" and not q_tokens:
        ws_part = [r for r in results if r["source"] == "workspace"][:6]
        sk_part = [r for r in results if r["source"] == "skill"][:4]
        rea_part = [r for r in results if r["source"] == "rea"][: max(1, cap - len(ws_part) - len(sk_part))]
        results = ws_part + sk_part + rea_part

    out: Dict[str, Any] = {
        "query": query,
        "source": src,
        "counts": counts,
        "tools": results[:cap],
    }
    if errors:
        out["warnings"] = errors
    return out


def inspect_tool(
    name: str,
    reference: str = "",
    workspace: Optional[Path] = None,
    builtin_help: Optional[Dict[str, Any]] = None,
    builtin_schemas: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return full parameter schema for a REA/built-in tool, or read a skill/workspace reference."""
    if not name:
        raise ValueError("name is required")

    if builtin_help and name in builtin_help:
        return {
            "name": name,
            "source": "builtin",
            "description": builtin_help[name],
            "schema": (builtin_schemas or {}).get(name),
        }

    # Check REA MCP tool schema
    try:
        schema = get_mcp_bridge().get_tool_schema(name)
        if schema:
            return {
                "name": name,
                "source": "rea",
                "description": schema.get("description", ""),
                "inputSchema": schema.get("inputSchema", {}),
            }
    except Exception as exc:
        mcp_err = str(exc)
    else:
        mcp_err = ""

    # Check skills
    for base in _skill_dirs(workspace):
        candidate = base / name
        if not candidate.is_dir():
            # also match by frontmatter name
            for child in base.iterdir():
                sm = child / "SKILL.md"
                if child.is_dir() and sm.is_file():
                    meta, _ = _parse_frontmatter(sm.read_text(encoding="utf-8", errors="replace"))
                    if meta.get("name") == name:
                        candidate = child
                        break
        skill_md = candidate / "SKILL.md"
        if skill_md.is_file():
            refs_dir = candidate / "references"
            avail_refs = sorted(p.name for p in refs_dir.glob("*.md")) if refs_dir.is_dir() else []
            target = refs_dir / reference if reference and refs_dir.is_dir() else skill_md
            if not target.is_file():
                raise ValueError(f"Skill reference not found: {reference!r}. Available: {avail_refs}")
            txt = target.read_text(encoding="utf-8", errors="replace")
            truncated = len(txt) > 6500
            return {
                "name": name,
                "source": "skill",
                "file": str(target),
                "available_references": avail_refs,
                "truncated": truncated,
                "content": txt[:6500],
            }

    # Check workspace script
    if workspace and workspace.is_dir():
        cand = (workspace / name)
        if not cand.is_file():
            for sub in ("tools", "waspo", "scripts"):
                p = workspace / sub / name
                if p.is_file():
                    cand = p
                    break
        if cand.is_file():
            lines = cand.read_text(encoding="utf-8", errors="replace").splitlines()
            return {
                "name": name,
                "source": "workspace",
                "file": str(cand),
                "line_count": len(lines),
                "preview": "\n".join(lines[:140]),
            }

    hint = f" (MCP status: {mcp_err})" if mcp_err else ""
    raise ValueError(f"Tool or skill {name!r} not found{hint}")


def extract_project_memory(workspace: Optional[Path], max_chars: int = 2600) -> str:
    """Read AGENTS.md / CLAUDE.md architectural rules + latest checkpoints for prompt injection."""
    if not workspace or not workspace.is_dir():
        return ""
    for fname in ("AGENTS.md", "CLAUDE.md"):
        p = workspace / fname
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                return ""
            if len(text) <= max_chars:
                return text
            head_budget = int(max_chars * 0.42)
            mid_budget = int(max_chars * 0.23)
            tail_budget = max(200, max_chars - head_budget - mid_budget - 90)
            head = text[:head_budget].rstrip()
            tail = text[-tail_budget:].lstrip()
            middle_slice = text[head_budget : max(head_budget, len(text) - tail_budget)]
            mid_lines: List[str] = []
            used = 0
            tool_rx = re.compile(r"(?i)(ghidra|mips|analyzeheadless|\brea\b|elf32|checkpoint)")
            head_rx = re.compile(r"^##+\s+.*")
            all_mid = middle_slice.splitlines()
            tool_hits = [ln[:180] for ln in all_mid if tool_rx.search(ln)]
            heading_hits = [ln[:120] for ln in all_mid if head_rx.search(ln) and not tool_rx.search(ln)]
            # Keep key RE/tooling/checkpoint lines plus recent section headings within budget
            for ln in tool_hits[:6] + tool_hits[-4:] + heading_hits[-6:]:
                if ln in mid_lines:
                    continue
                if used + len(ln) + 1 > mid_budget:
                    continue
                mid_lines.append(ln)
                used += len(ln) + 1
            mid = "\n".join(mid_lines)
            if mid:
                return f"{head}\n\n[... key middle headings & tooling notes from {fname} ...]\n{mid}\n\n[... latest tail of {fname} ...]\n{tail}"
            return f"{head}\n\n[... middle truncated; read {fname} for full log ...]\n\n{tail}"
    return ""


def _parse_pe_summary(data: bytes) -> Dict[str, Any]:
    if len(data) < 0x40 or data[:2] != b"MZ":
        return {}
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_off + 24 > len(data) or data[pe_off : pe_off + 4] != b"PE\x00\x00":
        return {"format": "MZ/DOS"}
    machine, num_sec, _ts, _sym, _nsym, opt_size, _chars = struct.unpack_from("<HHIIIHH", data, pe_off + 4)
    opt_off = pe_off + 24
    magic = struct.unpack_from("<H", data, opt_off)[0] if opt_off + 2 <= len(data) else 0
    is_pe32_plus = magic == 0x20B
    arch = {0x14C: "x86", 0x8664: "x64", 0x1C0: "ARM", 0xAA64: "ARM64"}.get(machine, hex(machine))
    sec_off = opt_off + opt_size
    sections = []
    for i in range(min(num_sec, 32)):
        off = sec_off + i * 40
        if off + 40 > len(data):
            break
        raw_name, vsize, vaddr, rsize, rptr, *_rest, s_chars = struct.unpack_from("<8sIIIIIIHI", data, off)
        name = raw_name.split(b"\x00", 1)[0].decode("ascii", errors="replace")
        sections.append({
            "name": name,
            "vaddr": hex(vaddr),
            "vsize": hex(vsize),
            "raw_offset": hex(rptr),
            "raw_size": hex(rsize),
        })
    return {
        "format": "PE32+" if is_pe32_plus else "PE32",
        "arch": arch,
        "sections": sections,
    }


def _parse_elf_summary(data: bytes) -> Dict[str, Any]:
    if len(data) < 52 or data[:4] != b"\x7fELF":
        return {}
    ei_class = data[4]
    ei_data = data[5]
    endian = "<" if ei_data == 1 else ">"
    bits = 32 if ei_class == 1 else 64
    e_type, e_machine = struct.unpack_from(endian + "HH", data, 16)
    entry = struct.unpack_from(endian + ("I" if bits == 32 else "Q"), data, 24)[0]
    mach_name = {3: "x86", 8: "MIPS", 62: "x86_64", 40: "ARM", 183: "AArch64"}.get(e_machine, hex(e_machine))
    return {
        "format": f"ELF{bits}",
        "endian": "little" if endian == "<" else "big",
        "machine": mach_name,
        "entry": hex(entry),
    }


def _parse_wad_summary(head: bytes) -> Optional[Dict[str, Any]]:
    if len(head) < 16 or head[:4] != b"WADH":
        return None
    data_offset, entry_count, names_size = struct.unpack_from("<3I", head, 4)
    str_off = 16 + entry_count * 24
    sample_entries: List[Dict[str, Any]] = []
    if 0 < entry_count <= 200_000 and str_off < len(head):
        str_end = min(len(head), str_off + names_size)
        for idx in range(entry_count):
            off = 16 + idx * 24
            if off + 24 > len(head):
                break
            name_off, data_rel_off, fsize, _flags, first_child, _next_sib = struct.unpack_from("<6i", head, off)
            name = ""
            if 0 <= name_off and str_off + name_off < str_end:
                nul = head.find(b"\x00", str_off + name_off, str_end)
                end_pos = nul if nul != -1 else str_end
                name = head[str_off + name_off : end_pos].decode("ascii", errors="replace")
            if not name and fsize <= 0 and first_child < 0:
                continue
            sample_entries.append({
                "index": idx,
                "name": name or f"entry_{idx}",
                "data_offset": (data_offset + data_rel_off) if fsize > 0 else 0,
                "size": fsize,
                "is_dir": bool(first_child >= 0 and fsize == 0),
            })
            if len(sample_entries) >= 20:
                break
    return {
        "format": "WADH_Archive",
        "header_magic": "WADH",
        "data_offset": data_offset,
        "entry_count": entry_count,
        "string_table_offset": str_off,
        "string_table_size": names_size,
        "sample_entries": sample_entries,
    }


def _extract_ascii_strings(data: bytes, min_len: int = 5, pattern: str = "", base_addr: int = 0, limit: int = 40) -> List[Dict[str, Any]]:
    rx = re.compile(rb"[\x20-\x7e]{%d,}" % max(3, min_len))
    pat_low = pattern.lower()
    hits: List[Dict[str, Any]] = []
    for m in rx.finditer(data):
        text = m.group().decode("ascii", errors="replace")
        if pat_low and pat_low not in text.lower():
            continue
        off = m.start()
        hits.append({"offset": hex(off), "vaddr": hex(base_addr + off), "text": text[:160]})
        if len(hits) >= limit:
            break
    return hits


def _scan_mips_xrefs(data: bytes, target_addr: int, base_addr: int = 0x100000, max_hits: int = 64) -> Dict[str, Any]:
    """Scan little-endian MIPS code/data for pointers, j/jal, and lui+low references."""
    hi = ((target_addr + 0x8000) >> 16) & 0xFFFF
    lo = target_addr & 0xFFFF
    lo_s = lo if lo < 0x8000 else lo - 0x10000
    j_field = (target_addr >> 2) & 0x03FFFFFF

    ptr_hits: List[str] = []
    jal_hits: List[str] = []
    j_hits: List[str] = []
    lui_hits: List[Dict[str, str]] = []

    n = len(data) - 3
    for off in range(0, n, 4):
        word = struct.unpack_from("<I", data, off)[0]
        va = base_addr + off
        if word == target_addr and len(ptr_hits) < max_hits:
            ptr_hits.append(hex(va))
        op = word >> 26
        if op == 3 and (word & 0x03FFFFFF) == j_field and len(jal_hits) < max_hits:
            jal_hits.append(hex(va))
        elif op == 2 and (word & 0x03FFFFFF) == j_field and len(j_hits) < max_hits:
            j_hits.append(hex(va))
        elif op == 0x0F and (word & 0xFFFF) == hi and len(lui_hits) < max_hits:
            rt = (word >> 16) & 0x1F
            # Check next 16 instructions for matching low-part user on register rt
            for delta in range(1, 17):
                off2 = off + delta * 4
                if off2 > n:
                    break
                w2 = struct.unpack_from("<I", data, off2)[0]
                op2 = w2 >> 26
                rs2 = (w2 >> 21) & 0x1F
                imm2 = w2 & 0xFFFF
                imm2_s = imm2 if imm2 < 0x8000 else imm2 - 0x10000
                # addiu (9), lw (0x23), sw (0x2B), lh/lhu/lb/lbu/sh/sb (0x20..0x29), ori (0xD)
                if rs2 == rt and ((op2 == 0x0D and imm2 == lo) or (op2 in (0x09, 0x20, 0x21, 0x23, 0x24, 0x25, 0x27, 0x28, 0x29, 0x2B, 0x37, 0x3F) and imm2_s == lo_s)):
                    lui_hits.append({"lui_va": hex(va), "use_va": hex(base_addr + off2), "op": hex(op2)})
                    break

    return {
        "target_addr": hex(target_addr),
        "base_addr": hex(base_addr),
        "pointer_hits": ptr_hits,
        "jal_hits": jal_hits,
        "j_hits": j_hits,
        "lui_pair_hits": lui_hits,
    }


def _parse_iso9660_geometry(path: Path) -> Optional[Tuple[int, int]]:
    """Return (sector_size, data_offset) if path is an ISO9660 image or raw 2352-byte Mode1/Mode2 BIN."""
    try:
        sz = path.stat().st_size
    except OSError:
        return None
    if sz < 17 * 2048:
        return None
    candidates = ((2352, 24), (2352, 16), (2048, 0))
    try:
        with path.open("rb") as f:
            for sec_sz, data_off in candidates:
                off = 16 * sec_sz + data_off
                if off + 8 > sz:
                    continue
                f.seek(off)
                pvd_head = f.read(8)
                if len(pvd_head) == 8 and pvd_head[:6] == b"\x01CD001":
                    return (sec_sz, data_off)
    except OSError:
        return None
    return None


def _read_iso_bytes(f, sec_sz: int, data_off: int, lba: int, size: int) -> bytes:
    if size <= 0:
        return b""
    n_sec = (size + 2047) // 2048
    parts = []
    for i in range(n_sec):
        f.seek((lba + i) * sec_sz + data_off)
        chunk = f.read(2048)
        if not chunk:
            break
        parts.append(chunk)
    return b"".join(parts)[:size]


def _walk_iso9660_entries(f, sec_sz: int, data_off: int, lba: int, size: int, prefix: str = "", depth: int = 0, out: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if out is None:
        out = []
    if depth > 4 or len(out) >= 512 or size <= 0 or size > 2_000_000:
        return out
    data = _read_iso_bytes(f, sec_sz, data_off, lba, size)
    pos = 0
    subdirs: List[Tuple[str, int, int]] = []
    while pos < len(data) and len(out) < 512:
        rlen = data[pos]
        if rlen == 0:
            nxt = ((pos // 2048) + 1) * 2048
            if nxt <= pos:
                break
            pos = nxt
            continue
        if pos + rlen > len(data) or rlen < 34:
            break
        rec = data[pos : pos + rlen]
        pos += rlen
        elba = struct.unpack_from("<I", rec, 2)[0]
        elen = struct.unpack_from("<I", rec, 10)[0]
        flags = rec[25]
        nlen = rec[32]
        if 33 + nlen > len(rec):
            continue
        raw_name = rec[33 : 33 + nlen]
        if raw_name in (b"\x00", b"\x01"):
            continue
        clean_name = raw_name.decode("ascii", errors="replace").split(";", 1)[0]
        if not clean_name:
            continue
        rel = f"{prefix}/{clean_name}" if prefix else clean_name
        is_dir = bool(flags & 2)
        out.append({"path": rel, "lba": elba, "size": elen, "is_dir": is_dir})
        if is_dir:
            subdirs.append((rel, elba, elen))
    # Recurse into subdirectories after listing all entries at the current level
    # so root-level files (SYSTEM.CNF, boot ELF, WADs) are never bumped by deep dirs.
    for rel, elba, elen in subdirs:
        if len(out) >= 512:
            break
        _walk_iso9660_entries(f, sec_sz, data_off, elba, elen, prefix=rel, depth=depth + 1, out=out)
    return out


def _parse_iso9660_summary(path: Path, include_all: bool = False) -> Optional[Dict[str, Any]]:
    geom = _parse_iso9660_geometry(path)
    if not geom:
        return None
    sec_sz, data_off = geom
    with path.open("rb") as f:
        pvd = _read_iso_bytes(f, sec_sz, data_off, 16, 2048)
        if len(pvd) < 190 or pvd[:6] != b"\x01CD001":
            return None
        vol_id = pvd[40:72].decode("ascii", errors="replace").strip()
        sys_id = pvd[8:40].decode("ascii", errors="replace").strip()
        root_rec = pvd[156:190]
        root_lba = struct.unpack_from("<I", root_rec, 2)[0]
        root_len = struct.unpack_from("<I", root_rec, 10)[0]
        entries = _walk_iso9660_entries(f, sec_sz, data_off, root_lba, root_len)
        files = sorted(
            (e for e in entries if not e["is_dir"]),
            key=lambda e: ("/" in e["path"], e["path"].lower()),
        )
        system_cnf = ""
        boot_exe = ""
        for e in files:
            if e["path"].upper() == "SYSTEM.CNF" and 0 < e["size"] <= 4096:
                system_cnf = _read_iso_bytes(f, sec_sz, data_off, e["lba"], e["size"]).decode("ascii", errors="replace").strip()
                m = re.search(r"BOOT2?\s*=\s*cdrom0?:[\\/]([^;\r\n\s]+)", system_cnf, re.I)
                if m:
                    boot_exe = m.group(1).strip()
                break
    mode_label = "ISO9660" if sec_sz == 2048 else ("ISO9660_Mode2_2352" if data_off == 24 else "ISO9660_Mode1_2352")
    return {
        "format": mode_label,
        "sector_size": sec_sz,
        "data_offset": data_off,
        "volume_id": vol_id,
        "system_id": sys_id,
        "boot_executable": boot_exe or None,
        "system_cnf": system_cnf or None,
        "iso_files": files if include_all else files[:60],
        "iso_file_count": len(files),
        "hint": (
            f"This .bin/.iso is a {mode_label} CD-ROM disc image, NOT raw executable code. "
            + (f"Main boot executable: {boot_exe}. " if boot_exe else "")
            + "Call binary_probe with mode='iso_extract' (optionally pattern='*' or a filename) "
            "to extract the executable and game files first, then run binary_probe / ghidra_analyze on the extracted files."
        ),
    }


def _extract_iso9660_files(path: Path, out_dir: Path, pattern: str = "") -> Dict[str, Any]:
    import fnmatch
    summary = _parse_iso9660_summary(path, include_all=True)
    if not summary:
        raise ValueError(f"'{path.name}' is not a recognized ISO9660 / 2352-byte CD-ROM image.")
    sec_sz = int(summary["sector_size"])
    data_off = int(summary["data_offset"])
    boot_exe = (summary.get("boot_executable") or "").lower()
    files = summary.get("iso_files") or []
    pat = (pattern or "").strip().lower()

    def _matches(entry: Dict[str, Any]) -> bool:
        entry_path = entry["path"]
        low = entry_path.lower()
        base = low.rsplit("/", 1)[-1]
        if not pat:
            return (
                base == "system.cnf"
                or (boot_exe and base == boot_exe)
                or bool(re.match(r"^(?:s[lca][eu]s_\d+\.\d+|.*\.elf)$", base))
                or ("/" not in low and entry["size"] <= 100_000_000 and not low.endswith((".url", ".cue")))
            )
        return pat in low or fnmatch.fnmatch(low, pat) or fnmatch.fnmatch(base, pat)

    selected = [e for e in files if _matches(e)]
    if not selected and not pat:
        selected = [e for e in files if e["size"] <= 20_000_000][:10]
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with path.open("rb") as f:
        for e in selected:
            if e["size"] > 100_000_000:
                extracted.append({"path": e["path"], "skipped": f"too large ({e['size']} bytes); extract specifically by filename if needed"})
                continue
            data = _read_iso_bytes(f, sec_sz, data_off, int(e["lba"]), int(e["size"]))
            dest = (out_dir / e["path"]).resolve()
            if out_dir.resolve() not in dest.parents and dest != out_dir.resolve():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            rec: Dict[str, Any] = {
                "iso_path": e["path"],
                "extracted_path": str(dest),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            elf = _parse_elf_summary(data)
            if elf:
                rec.update(elf)
            else:
                wad = _parse_wad_summary(data[:65536])
                if wad:
                    rec.update(wad)
            extracted.append(rec)
    return {
        "path": str(path),
        "format": summary["format"],
        "boot_executable": summary.get("boot_executable"),
        "output_dir": str(out_dir),
        "extracted_count": sum(1 for x in extracted if "extracted_path" in x),
        "extracted": extracted,
    }


def binary_probe(
    path: Path,
    mode: str = "info",
    offset: Any = 0,
    length: int = 256,
    target_addr: Any = None,
    base_addr: Any = 0x100000,
    pattern: str = "",
    endian: str = "little",
    max_hits: int = 64,
) -> Dict[str, Any]:
    """Fast local binary inspector: ISO9660/BIN extract, headers, hex dump, string search, pointer & MIPS R5900 xref scan."""
    if not path.is_file():
        raise FileNotFoundError(f"Binary not found: {path}")
    size = path.stat().st_size
    m = (mode or "info").strip().lower()
    base = _parse_int(base_addr, 0x100000)
    off = max(0, _parse_int(offset, 0))
    cap = max(1, min(200, int(max_hits or 64)))

    if m == "iso_extract":
        out_dir = path.parent / f"{path.stem}_extracted"
        return _extract_iso9660_files(path, out_dir, pattern=pattern)

    if m == "info":
        h = hashlib.sha256()
        with path.open("rb") as f:
            head = f.read(65536)
            h.update(head)
            while True:
                blk = f.read(1048576)
                if not blk:
                    break
                h.update(blk)
        info: Dict[str, Any] = {
            "path": str(path),
            "size": size,
            "sha256": h.hexdigest(),
            "magic": head[:16].hex(),
        }
        iso = _parse_iso9660_summary(path)
        pe = _parse_pe_summary(head)
        elf = _parse_elf_summary(head)
        wad = _parse_wad_summary(head)
        if iso:
            info.update(iso)
        elif pe:
            info.update(pe)
        elif elf:
            info.update(elf)
        elif wad:
            info.update(wad)
        else:
            info["format"] = "raw"
        if not iso:
            head_strs = _extract_ascii_strings(head, min_len=6, base_addr=base, limit=80)
            id_strs = [s for s in head_strs if re.search(r"[A-Za-z]{4,}", s["text"])]
            if len(id_strs) < 5 and size > len(head):
                with path.open("rb") as f:
                    scan_buf = f.read(min(size, 4 * 1024 * 1024))
                full_strs = _extract_ascii_strings(scan_buf, min_len=6, base_addr=base, limit=200)
                id_strs = [s for s in full_strs if re.search(r"[A-Za-z]{4,}", s["text"])]
                other_strs = [s for s in full_strs if s not in id_strs]
            else:
                other_strs = [s for s in head_strs if s not in id_strs]
            info["sample_strings"] = (id_strs + other_strs)[:15]
        return info

    if m == "hex":
        nbytes = max(16, min(2048, int(length or 256)))
        with path.open("rb") as f:
            f.seek(off)
            chunk = f.read(nbytes)
        lines = []
        for i in range(0, len(chunk), 16):
            row = chunk[i : i + 16]
            hx = " ".join(f"{b:02x}" for b in row)
            asc = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
            lines.append(f"{off + i:08x} ({base + off + i:08x}): {hx:<47}  |{asc}|")
        return {
            "path": str(path),
            "offset": hex(off),
            "base_addr": hex(base),
            "bytes_shown": len(chunk),
            "dump": "\n".join(lines),
        }

    with path.open("rb") as f:
        raw = f.read(min(size, 64 * 1024 * 1024))

    if m == "strings":
        return {
            "path": str(path),
            "pattern": pattern,
            "hits": _extract_ascii_strings(raw, min_len=5, pattern=pattern, base_addr=base, limit=cap),
        }

    if m == "scan_u32":
        if target_addr is None or target_addr == "":
            raise ValueError("target_addr is required for scan_u32 mode")
        val = _parse_int(target_addr) & 0xFFFFFFFF
        needle = struct.pack("<I" if endian != "big" else ">I", val)
        hits = []
        pos = 0
        while len(hits) < cap:
            idx = raw.find(needle, pos)
            if idx == -1:
                break
            hits.append({"offset": hex(idx), "vaddr": hex(base + idx)})
            pos = idx + 1
        return {"path": str(path), "target": hex(val), "endian": endian, "hits": hits}

    if m == "mips_xrefs":
        if target_addr is None or target_addr == "":
            raise ValueError("target_addr is required for mips_xrefs mode")
        val = _parse_int(target_addr) & 0xFFFFFFFF
        res = _scan_mips_xrefs(raw, val, base_addr=base, max_hits=cap)
        res["path"] = str(path)
        return res

    raise ValueError(f"Unknown binary_probe mode {mode!r}; expected info, hex, strings, scan_u32, mips_xrefs, or iso_extract")


def _wrap_raw_mips_elf32(raw: bytes, vaddr: int, entry: int) -> bytes:
    """Wrap raw MIPS R5900 binary in a minimal ELF32 ET_EXEC with 1 PT_LOAD at vaddr."""
    ehdr_size = 52
    phdr_size = 32
    offset = 0x1000
    ehdr = struct.pack(
        "<16sHHIIIIIHHHHHH",
        b"\x7fELF\x01\x01\x01" + b"\x00" * 9,
        2,              # ET_EXEC
        8,              # EM_MIPS
        1,              # EV_CURRENT
        entry & 0xFFFFFFFF,
        ehdr_size,      # e_phoff
        0,              # e_shoff
        0x20000000,     # EF_MIPS_ARCH_3
        ehdr_size,
        phdr_size,
        1,              # e_phnum
        0, 0, 0,
    )
    phdr = struct.pack(
        "<IIIIIIII",
        1,              # PT_LOAD
        offset,
        vaddr & 0xFFFFFFFF,
        vaddr & 0xFFFFFFFF,
        len(raw),
        len(raw),
        7,              # PF_R | PF_W | PF_X
        0x1000,
    )
    pad = b"\x00" * (offset - ehdr_size - phdr_size)
    return ehdr + phdr + pad + raw


def ghidra_analyze(
    binary_path: Path,
    processor: str = "MIPS:LE:64:64-32addr",
    base_addr: Any = None,
    wrap_elf: bool = False,
    elf_entry: Any = None,
    post_script: str = "",
    script_path: Optional[Path] = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Run Ghidra analyzeHeadless cleanly on an isolated temporary copy (supports MIPS ELF wrapping)."""
    if not binary_path.is_file():
        raise FileNotFoundError(f"Binary not found: {binary_path}")

    iso = _parse_iso9660_summary(binary_path)
    if iso is not None:
        boot = iso.get("boot_executable") or "the main executable"
        return {
            "ok": False,
            "status": "failed",
            "error": (
                f"Target '{binary_path.name}' is a {iso['format']} CD-ROM disc image, not a single executable. "
                f"Call binary_probe(path=..., mode='iso_extract') first to extract {boot}, then run ghidra_analyze on the extracted ELF."
            ),
        }

    ghidra_dir = Path(os.environ.get("GHIDRA_INSTALL_DIR") or DEFAULT_GHIDRA_DIR)
    headless = ghidra_dir / "support" / "analyzeHeadless.bat"
    if not headless.is_file():
        raise FileNotFoundError(f"Ghidra analyzeHeadless.bat not found at {headless}")

    env = dict(os.environ)
    env["GHIDRA_INSTALL_DIR"] = str(ghidra_dir)
    j21 = _find_java21_home()
    if j21:
        env["JAVA_HOME"] = j21
        env["PATH"] = str(Path(j21) / "bin") + os.pathsep + env.get("PATH", "")

    raw_bytes = binary_path.read_bytes()
    already_elf = raw_bytes[:4] == b"\x7fELF"
    elf_info = _parse_elf_summary(raw_bytes[:64]) if already_elf else {}
    effective_wrap = bool(wrap_elf and not already_elf)

    with tempfile.TemporaryDirectory(prefix="jarvis_ghidra_") as tmp:
        tmp_dir = Path(tmp)
        proj_dir = tmp_dir / "proj"
        proj_dir.mkdir(parents=True, exist_ok=True)
        import_path = tmp_dir / binary_path.name
        if effective_wrap:
            vaddr = _parse_int(base_addr, 0x100000)
            entry = _parse_int(elf_entry, vaddr)
            wrapped_bytes = _wrap_raw_mips_elf32(raw_bytes, vaddr=vaddr, entry=entry)
            import_path = tmp_dir / (binary_path.stem + "_wrapped.elf")
            import_path.write_bytes(wrapped_bytes)
        else:
            import_path.write_bytes(raw_bytes)

        # Invoke the .bat directly (without "cmd.exe /c") so Windows CreateProcess
        # properly quotes both headless.bat and path arguments containing spaces.
        cmd = [
            str(headless),
            str(proj_dir), "JarvisProj",
            "-import", str(import_path),
            "-processor", processor,
            "-deleteProject",
        ]
        if not effective_wrap and not already_elf and base_addr is not None and str(base_addr).strip() != "":
            b_hex = hex(_parse_int(base_addr))
            cmd.extend(["-loader", "BinaryLoader", "-loader-baseAddr", b_hex])
        if script_path and Path(script_path).exists():
            cmd.extend(["-scriptPath", str(script_path)])
        if post_script:
            cmd.extend(["-postScript", post_script])

        proc = subprocess.run(
            cmd,
            cwd=str(headless.parent),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(30, min(600, int(timeout or 180))),
        )
        resolved_base = _parse_int(base_addr, 0x100000) if base_addr is not None else 0x100000
        out: Dict[str, Any] = {
            "ok": proc.returncode == 0,
            "status": "completed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "processor": processor,
            "wrapped_elf": effective_wrap,
            "base_addr": hex(_parse_int(base_addr)) if base_addr is not None else None,
            "elf_info": elf_info or None,
            "sample_strings": _extract_ascii_strings(raw_bytes, min_len=5, base_addr=resolved_base, limit=25),
            "stdout_tail": (proc.stdout or "")[-3500:],
            "stderr_tail": (proc.stderr or "")[-1500:],
        }
        if proc.returncode != 0:
            out["error"] = (proc.stderr or proc.stdout or f"Ghidra exited with code {proc.returncode}")[-600:]
        return out
