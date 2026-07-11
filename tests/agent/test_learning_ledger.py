import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from agent import learning_ledger as ll


def _agent(**kw):
    base = {
        "session_id": "sess-1",
        "task_id": "task-1",
        "platform": "cli",
        "provider": "openai",
        "model": "gpt-test",
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_build_turn_record_extracts_learning_signals():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"cmd":"pytest"}'}},
                {"function": {"name": "skill_manage", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "name": "terminal", "content": "passed"},
    ]

    rec = ll.build_turn_record(
        _agent(),
        user_message="Please fix this failing test",
        final_response="Fixed and verified.",
        messages=messages,
        api_call_count=3,
        interrupted=False,
        failed=False,
        turn_exit_reason="completed",
    )

    assert rec["schema_version"] == 1
    assert rec["outcome"] == "completed"
    assert rec["task_type"] == "skill-work"
    assert rec["tool_counts"]["terminal"] == 1
    assert rec["tool_counts"]["skill_manage"] == 1
    assert rec["learning_signals"]["used_skill_tool"] is True
    assert rec["learning_signals"]["used_terminal"] is True
    assert rec["tool_result_count"] == 1
    assert rec["tool_error_count"] == 0


def test_build_turn_record_redacts_secret_previews():
    rec = ll.build_turn_record(
        _agent(),
        user_message="OPENAI_API_KEY=sk-testsecret1234567890",
        final_response="password: hunter2",
        messages=[],
        api_call_count=1,
        interrupted=False,
        failed=False,
    )

    assert "testsecret" not in rec["user_preview"]
    assert "hunter2" not in rec["assistant_preview"]


def test_build_turn_record_counts_only_real_tool_errors():
    rec = ll.build_turn_record(
        _agent(),
        user_message="Run tools",
        final_response="Done.",
        messages=[
            {"role": "tool", "content": '{"success": true, "error": null}'},
            {"role": "tool", "content": '{"success": false, "error": "failed"}'},
        ],
        api_call_count=1,
        interrupted=False,
        failed=False,
    )

    assert rec["tool_result_count"] == 2
    assert rec["tool_error_count"] == 1


def test_learning_config_is_a_recognized_root_section():
    from hermes_cli.config import validate_config_structure

    assert validate_config_structure({"learning": {"ledger_enabled": False}}) == []


def test_build_turn_record_bounds_previews():
    rec = ll.build_turn_record(
        _agent(),
        user_message="x" * 1000,
        final_response=[{"type": "text", "text": "y" * 1000}],
        messages=[],
        api_call_count=1,
        interrupted=False,
        failed=False,
    )

    assert len(rec["user_preview"]) <= 320
    assert len(rec["assistant_preview"]) <= 320
    assert rec["user_preview"].endswith("...")
    assert rec["assistant_preview"].endswith("...")


def test_record_turn_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(ll, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(ll, "_config", lambda: (True, 5_000_000))

    ok = ll.record_turn(
        _agent(),
        user_message="Review this code",
        final_response="No issues found.",
        messages=[],
        api_call_count=1,
        interrupted=False,
        failed=False,
    )

    assert ok is True
    path = tmp_path / "learning" / "turns.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["task_type"] == "review"
    assert rec["session_id"] == "sess-1"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_record_turn_skips_persistence_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(ll, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(ll, "_config", lambda: (True, 5_000_000))

    ok = ll.record_turn(
        _agent(_persist_disabled=True),
        user_message="Review this code",
        final_response="Done.",
        messages=[],
        api_call_count=1,
        interrupted=False,
        failed=False,
    )

    assert ok is False
    assert not (tmp_path / "learning" / "turns.jsonl").exists()


def test_record_turn_respects_disabled_config(tmp_path, monkeypatch):
    monkeypatch.setattr(ll, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(ll, "_config", lambda: (False, 5_000_000))

    ok = ll.record_turn(
        _agent(),
        user_message="Review this code",
        final_response="Done.",
        messages=[],
        api_call_count=1,
        interrupted=False,
        failed=False,
    )

    assert ok is False
    assert not (tmp_path / "learning" / "turns.jsonl").exists()


def test_record_turn_serializes_concurrent_writers(tmp_path, monkeypatch):
    monkeypatch.setattr(ll, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(ll, "_config", lambda: (True, 5_000_000))

    def write(index):
        return ll.record_turn(
            _agent(task_id=f"task-{index}"),
            user_message=f"Review item {index}",
            final_response="Done.",
            messages=[],
            api_call_count=1,
            interrupted=False,
            failed=False,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(write, range(40)))

    path = tmp_path / "learning" / "turns.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 40
    assert {record["task_id"] for record in records} == {f"task-{index}" for index in range(40)}


def test_enforce_max_bytes_keeps_recent_complete_lines(tmp_path):
    path = tmp_path / "turns.jsonl"
    path.write_text('{"old":1}\n{"new":2}\n', encoding="utf-8")

    ll._enforce_max_bytes(path, len('{"new":2}\n'))

    assert path.read_text(encoding="utf-8") == '{"new":2}\n'
