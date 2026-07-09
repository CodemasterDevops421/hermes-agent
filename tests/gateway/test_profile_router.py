"""Tests for conservative content-based profile routing."""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig
from gateway.profile_router import (
    ProfileRoutingCandidate,
    ProfileStickiness,
    conversation_key,
    select_profile_for_message,
)
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

    def test_keeps_code_work_on_default_for_delegation(self):
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
        assert result == "default"

    def test_keeps_apify_actor_code_review_on_default_for_delegation(self):
        candidates = [
            ProfileRoutingCandidate("default", "General-purpose Hermes profile."),
            ProfileRoutingCandidate("coding", "Codebase inspection and debugging."),
            ProfileRoutingCandidate("devops", "Deployments, Docker, and servers."),
        ]
        result = select_profile_for_message(
            "Can you review this apify actor code. Apify automatic checks are getting failed",
            candidates,
            current_profile="default",
        )
        assert result == "default"

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


class TestProfileStickiness:
    def test_pin_and_get(self):
        clock = [100.0]
        s = ProfileStickiness(ttl_seconds=900, clock=lambda: clock[0])
        key = ("telegram", "123", "", "42")
        s.pin(key, "coding")
        assert s.get(key) == "coding"

    def test_pin_expires_after_ttl(self):
        clock = [100.0]
        s = ProfileStickiness(ttl_seconds=900, clock=lambda: clock[0])
        key = ("telegram", "123", "", "42")
        s.pin(key, "coding")
        clock[0] += 901
        assert s.get(key) is None

    def test_refresh_extends_ttl(self):
        clock = [100.0]
        s = ProfileStickiness(ttl_seconds=900, clock=lambda: clock[0])
        key = ("telegram", "123", "", "42")
        s.pin(key, "coding")
        clock[0] += 800
        s.refresh(key)
        clock[0] += 800  # 1600 past pin, but only 800 past refresh
        assert s.get(key) == "coding"

    def test_repin_overrides_previous_profile(self):
        clock = [100.0]
        s = ProfileStickiness(ttl_seconds=900, clock=lambda: clock[0])
        key = ("telegram", "123", "", "42")
        s.pin(key, "coding")
        s.pin(key, "devops")
        assert s.get(key) == "devops"

    def test_conversation_key_requires_chat_id(self):
        class _Src:
            platform = None
            chat_id = None
            thread_id = None
            user_id = None

        assert conversation_key(_Src()) is None
        assert conversation_key(None) is None

        src = _Src()
        src.chat_id = "123"
        src.user_id = "42"
        assert conversation_key(src) == ("None", "123", "", "42")


class TestGatewayProfileHandler:
    @pytest.mark.asyncio
    async def test_default_handler_keeps_code_work_on_default(self):
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
        assert seen["profile"] == "default"

    @pytest.mark.asyncio
    async def test_follow_up_sticks_to_routed_ops_profile(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_router_candidates = [
            ProfileRoutingCandidate("default", "General-purpose Hermes profile."),
            ProfileRoutingCandidate("coding", "Codebase inspection and debugging."),
            ProfileRoutingCandidate("devops", "Deployments, Docker, and servers."),
        ]

        seen = []

        async def _fake_handle(event):
            seen.append(event.source.profile)
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("default")

        def _make_event(text, chat_id="123"):
            class _Src:
                profile = None
                platform = None
                thread_id = None
                user_id = "42"

            class _Evt:
                source = _Src()

            _Evt.source.chat_id = chat_id
            _Evt.text = text
            return _Evt()

        await handler(_make_event("fix the s3 backup and disk pressure on the pi"))
        await handler(_make_event("thanks, now what should I do?"))
        await handler(_make_event("hello there", chat_id="other-chat"))
        assert seen == ["devops", "devops", "default"]

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
