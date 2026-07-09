"""Tests for default-profile scoping of the specialist delegation guidance."""

from __future__ import annotations

from pathlib import Path

import hermes_cli.profiles
from agent.system_prompt import _is_default_profile


class TestIsDefaultProfile:
    def test_true_for_default_home(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        assert _is_default_profile() is True

    def test_false_for_profile_home(self, monkeypatch):
        profile_home = Path.home() / ".hermes" / "profiles" / "coding"
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        assert _is_default_profile() is False

    def test_fails_open_on_resolution_error(self, monkeypatch):
        def _boom() -> str:
            raise RuntimeError("resolution failed")

        monkeypatch.setattr(
            hermes_cli.profiles, "get_active_profile_name", _boom
        )
        assert _is_default_profile() is True
