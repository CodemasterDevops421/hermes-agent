"""Passive learning-evidence ledger for completed agent turns.

The memory and skill systems decide what to retain. This ledger records the
evidence those decisions should later be judged against: what kind of turn ran,
which tools were used, whether the turn completed, and small bounded previews of
the user request and final response.

It is deliberately non-blocking and non-prescriptive. It does not call a model,
does not alter prompts, and does not promote skills. Future evaluators can read
the JSONL records and decide which lessons deserve validation or rollback.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_SCHEMA_VERSION = 1
_MAX_PREVIEW_CHARS = 320
_MAX_TOOL_NAMES = 40
_DEFAULT_MAX_BYTES = 5_000_000
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?im)\b(api[_ -]?key|token|secret|password|passwd|credential|authorization)"
    r"\s*[:=]\s*([^\s,;]+)"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ledger_path() -> Path:
    return get_hermes_home() / "learning" / "turns.jsonl"


@contextmanager
def _ledger_lock(path: Path):
    """Serialize append and rotation across concurrent gateway turns."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.chmod(lock_path, 0o600)
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _config() -> tuple[bool, int]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        learning = cfg.get("learning", {}) if isinstance(cfg, dict) else {}
        if not isinstance(learning, dict):
            learning = {}
        enabled = bool(learning.get("ledger_enabled", True))
        try:
            max_bytes = int(learning.get("ledger_max_bytes", _DEFAULT_MAX_BYTES))
        except (TypeError, ValueError):
            max_bytes = _DEFAULT_MAX_BYTES
        return enabled, max(0, max_bytes)
    except Exception:
        return True, _DEFAULT_MAX_BYTES


def _compact_text(value: Any, *, limit: int = _MAX_PREVIEW_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"}:
                    parts.append(str(item.get("text") or ""))
                elif item.get("text"):
                    parts.append(str(item.get("text") or ""))
            elif item is not None:
                parts.append(str(item))
        text = "\n".join(p for p in parts if p)
    else:
        text = str(value)
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text, force=True, file_read=True)
        text = _CREDENTIAL_ASSIGNMENT_RE.sub(r"\1: «redacted-secret»", text)
    except Exception:
        # The ledger is optional telemetry; redactor import failure must not
        # affect the foreground turn.
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _tool_name_from_call(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    fn = call.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name") or "").strip()
    return str(call.get("name") or "").strip()


def _extract_tool_names(messages: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            name = _tool_name_from_call(call)
            if name:
                names.append(name)
    return names


def _tool_result_counts(messages: list[dict[str, Any]] | None) -> tuple[int, int]:
    total = 0
    errors = 0
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        total += 1
        content = str(msg.get("content") or "")
        lowered = content.lower()
        is_error = lowered.startswith("error")
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                is_error = parsed.get("success") is False or bool(parsed.get("error"))
        except (TypeError, ValueError):
            pass
        if is_error:
            errors += 1
    return total, errors


def _classify_task(user_preview: str, tool_counts: Counter[str]) -> str:
    text = user_preview.lower()
    if any(name in tool_counts for name in ("skill_manage", "skill_view", "skills_list")):
        return "skill-work"
    if any(name in tool_counts for name in ("memory", "session_search")):
        return "memory-work"
    if any(word in text for word in ("bug", "error", "fail", "traceback", "debug", "fix")):
        return "debugging"
    if any(word in text for word in ("review", "audit", "inspect")):
        return "review"
    if any(word in text for word in ("implement", "add", "build", "change", "update")):
        return "implementation"
    if any(word in text for word in ("explain", "why", "how", "what")):
        return "explanation"
    return "general"


def _outcome(*, final_response: Any, interrupted: bool, failed: bool) -> str:
    if interrupted:
        return "interrupted"
    if failed:
        return "failed"
    if not _compact_text(final_response):
        return "empty"
    return "completed"


def build_turn_record(
    agent: Any,
    *,
    user_message: Any,
    final_response: Any,
    messages: list[dict[str, Any]] | None,
    api_call_count: int,
    interrupted: bool,
    failed: bool,
    turn_exit_reason: str = "",
) -> dict[str, Any]:
    """Return a bounded, JSON-serializable learning evidence record."""

    tool_names = _extract_tool_names(messages)
    tool_counts = Counter(tool_names)
    tool_result_count, tool_error_count = _tool_result_counts(messages)
    user_preview = _compact_text(user_message)
    final_preview = _compact_text(final_response)
    outcome = _outcome(final_response=final_response, interrupted=interrupted, failed=failed)

    return {
        "schema_version": _SCHEMA_VERSION,
        "created_at": _utc_now(),
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "task_id": str(getattr(agent, "task_id", "") or ""),
        "platform": str(getattr(agent, "platform", "") or "cli"),
        "provider": str(getattr(agent, "provider", "") or ""),
        "model": str(getattr(agent, "model", "") or ""),
        "outcome": outcome,
        "turn_exit_reason": str(turn_exit_reason or ""),
        "api_call_count": int(api_call_count or 0),
        "task_type": _classify_task(user_preview, tool_counts),
        "user_preview": user_preview,
        "assistant_preview": final_preview,
        "tool_call_count": len(tool_names),
        "tool_result_count": tool_result_count,
        "tool_error_count": tool_error_count,
        "tools_used": sorted(tool_counts)[:_MAX_TOOL_NAMES],
        "tool_counts": dict(sorted(tool_counts.items())),
        "learning_signals": {
            "used_memory_tool": "memory" in tool_counts,
            "used_skill_tool": any(name.startswith("skill") or name == "skills_list" for name in tool_counts),
            "used_terminal": "terminal" in tool_counts,
            "used_delegation": "delegate_task" in tool_counts,
        },
    }


def record_turn(
    agent: Any,
    *,
    user_message: Any,
    final_response: Any,
    messages: list[dict[str, Any]] | None,
    api_call_count: int,
    interrupted: bool,
    failed: bool,
    turn_exit_reason: str = "",
) -> bool:
    """Append one turn record. Returns False on any best-effort failure."""

    try:
        if getattr(agent, "_persist_disabled", False):
            return False
        enabled, max_bytes = _config()
        if not enabled:
            return False
        record = build_turn_record(
            agent,
            user_message=user_message,
            final_response=final_response,
            messages=messages,
            api_call_count=api_call_count,
            interrupted=interrupted,
            failed=failed,
            turn_exit_reason=turn_exit_reason,
        )
        path = _ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        with _ledger_lock(path):
            fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                os.chmod(path, 0o600)
                os.write(fd, payload)
            finally:
                os.close(fd)
            _enforce_max_bytes(path, max_bytes)
        return True
    except Exception:
        return False


def _enforce_max_bytes(path: Path, max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    try:
        if path.stat().st_size <= max_bytes:
            return
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        kept: list[str] = []
        total = 0
        for line in reversed(lines):
            line_bytes = len(line.encode("utf-8")) + 1
            if kept and total + line_bytes > max_bytes:
                break
            kept.append(line)
            total += line_bytes
        kept.reverse()
        replacement = path.with_suffix(path.suffix + ".tmp")
        replacement.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
        os.chmod(replacement, 0o600)
        os.replace(replacement, path)
    except Exception:
        return
