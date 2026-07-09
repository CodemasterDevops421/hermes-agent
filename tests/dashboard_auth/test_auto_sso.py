"""Regression tests for auto-SSO provider selection.

A password-only provider (BasicAuthProvider) has no OAuth redirect flow, so
the single-provider auto-SSO shortcut must fall through to the /login
credential form instead of bouncing to /auth/login (which raises
NotImplementedError → 500 for password providers).
"""

from __future__ import annotations

from starlette.requests import Request

import hermes_cli.dashboard_auth.middleware as mw


def _request(path: str = "/") -> Request:
    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"dash.example.com"), (b"accept", b"text/html")],
        "server": ("dash.example.com", 80),
        "client": ("203.0.113.9", 4242),
    })


class _Provider:
    def __init__(self, name: str, supports_password: bool):
        self.name = name
        self.supports_password = supports_password


class TestAutoSsoProviderSelection:
    def test_password_only_provider_never_auto_redirects(self, monkeypatch):
        monkeypatch.setattr(
            mw, "list_session_providers",
            lambda: [_Provider("basic", supports_password=True)],
        )
        assert mw._auto_sso_response(_request()) is None

    def test_single_oauth_provider_still_redirects(self, monkeypatch):
        monkeypatch.setattr(
            mw, "list_session_providers",
            lambda: [_Provider("nous", supports_password=False)],
        )
        resp = mw._auto_sso_response(_request())
        assert resp is not None
        assert resp.status_code == 302
        assert "/auth/login?provider=nous" in resp.headers["location"]
