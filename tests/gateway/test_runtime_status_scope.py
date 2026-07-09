"""Secondary-profile adapters must not clobber the shared runtime status.

The runtime status file is keyed by platform only. In multiplex mode a
token-less profile adapter that fails to connect used to stamp
``telegram: disconnected`` over the primary adapter's healthy state.
"""

from __future__ import annotations

import gateway.status
from gateway.platforms.base import BasePlatformAdapter


class _Platform:
    value = "telegram"


class _StubAdapter(BasePlatformAdapter):
    async def connect(self):  # pragma: no cover - unused stubs
        raise NotImplementedError

    async def disconnect(self):  # pragma: no cover
        raise NotImplementedError

    async def get_chat_info(self, chat_id):  # pragma: no cover
        raise NotImplementedError

    async def send(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError


def _adapter(suppress: bool) -> BasePlatformAdapter:
    a = object.__new__(_StubAdapter)
    a.platform = _Platform()
    if suppress:
        a._suppress_shared_runtime_status = True
    return a


class TestRuntimeStatusScope:
    def test_secondary_profile_adapter_never_writes_shared_status(self, monkeypatch):
        written = []
        monkeypatch.setattr(
            gateway.status, "write_runtime_status",
            lambda **kw: written.append(kw),
        )
        _adapter(suppress=True)._write_runtime_status_safe(
            "disconnected", platform_state="disconnected",
            error_code=None, error_message=None,
        )
        assert written == []

    def test_primary_adapter_still_writes_status(self, monkeypatch):
        written = []
        monkeypatch.setattr(
            gateway.status, "write_runtime_status",
            lambda **kw: written.append(kw),
        )
        _adapter(suppress=False)._write_runtime_status_safe(
            "connected", platform_state="connected",
            error_code=None, error_message=None,
        )
        assert written and written[0]["platform"] == "telegram"
        assert written[0]["platform_state"] == "connected"
