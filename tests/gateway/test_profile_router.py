"""Tests for conservative content-based profile routing."""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig
from gateway.profile_router import ProfileRoutingCandidate, select_profile_for_message
from gateway.run import GatewayRunner


class TestProfileRouter:
    def test_keeps_default_for_ordinary_chat(self):
        candidates = [
            ProfileRoutingCandidate("default", "General-purpose Hermes profile."),
            ProfileRoutingCandidate("coding", "Codebase inspection and debugging."),
            ProfileRoutingCandidate("research", "Research and analysis."),
        ]
        result = select_profile_for_message(
            "did we already discuss the spreadsheet I created yesterday?",
            candidates,
            current_profile="default",
        )
        assert result == "default"

    def test_routes_code_work_to_coding(self):
        candidates = [
            ProfileRoutingCandidate("default", "General-purpose Hermes profile."),
            ProfileRoutingCandidate("coding", "Codebase inspection and debugging."),
            ProfileRoutingCandidate("devops", "Deployments, Docker, and servers."),
        ]
        result = select_profile_for_message(
            "please debug this failing python test and refactor the function",
            candidates,
            current_profile="default",
        )
        assert result == "coding"

    def test_routes_ops_work_to_devops(self):
        candidates = [
            ProfileRoutingCandidate("default", "General-purpose Hermes profile."),
            ProfileRoutingCandidate("coding", "Codebase inspection and debugging."),
            ProfileRoutingCandidate("devops", "Deployments, Docker, and servers."),
        ]
        result = select_profile_for_message(
            "the pi is under disk pressure, fix the backup and s3 sync",
            candidates,
            current_profile="default",
        )
        assert result == "devops"

    def test_named_profile_is_preserved(self):
        candidates = [
            ProfileRoutingCandidate("default", "General-purpose Hermes profile."),
            ProfileRoutingCandidate("coding", "Codebase inspection and debugging."),
        ]
        result = select_profile_for_message(
            "please debug this failing test",
            candidates,
            current_profile="research",
        )
        assert result == "research"


class TestGatewayProfileHandler:
    @pytest.mark.asyncio
    async def test_default_handler_can_reroute_to_coding(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_router_candidates = [
            ProfileRoutingCandidate("default", "General-purpose Hermes profile."),
            ProfileRoutingCandidate("coding", "Codebase inspection and debugging."),
            ProfileRoutingCandidate("research", "Research and analysis."),
        ]

        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("default")

        class _Src:
            profile = None

        class _Evt:
            source = _Src()
            text = "please debug this failing test"

        result = await handler(_Evt())
        assert result == "ok"
        assert seen["profile"] == "coding"

    @pytest.mark.asyncio
    async def test_existing_named_profile_stays_put(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_router_candidates = [
            ProfileRoutingCandidate("default", "General-purpose Hermes profile."),
            ProfileRoutingCandidate("coding", "Codebase inspection and debugging."),
        ]

        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("research")

        class _Src:
            profile = None

        class _Evt:
            source = _Src()
            text = "please debug this failing test"

        await handler(_Evt())
        assert seen["profile"] == "research"
