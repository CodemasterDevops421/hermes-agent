"""Conservative content-based routing for multiplexed Hermes profiles.

The router only overrides the default profile when the message contains
strong specialist signals. Ordinary chat stays on the default profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Callable, Iterable, Optional, Sequence

from hermes_cli.profiles import profiles_to_serve, read_profile_meta


@dataclass(frozen=True)
class ProfileRoutingCandidate:
    name: str
    description: str = ""
    is_default: bool = False


_TOKEN_RE = re.compile(r"[a-z0-9]+")

_PROFILE_HINTS: dict[str, tuple[str, ...]] = {
    "coding": (
        "code", "coding", "debug", "bug", "bugs", "error", "errors", "test",
        "tests", "pytest", "refactor", "lint", "build", "compile", "repo",
        "github", "function", "method", "class", "python", "javascript",
        "typescript", "sql", "script", "patch", "diff", "commit", "branch",
        "pull request", "unit test",
    ),
    "devops": (
        "deploy", "deployment", "docker", "container", "containers",
        "kubernetes", "k8s", "terraform", "aws", "s3", "backup", "disk",
        "storage", "server", "daemon", "systemd", "cron", "network", "port",
        "infrastructure", "ops", "prune", "disk pressure", "memory pressure",
        "restore", "snapshot",
    ),
    "research": (
        "research", "audit", "analyze", "analysis", "compare", "benchmark",
        "paper", "papers", "report", "study", "findings", "evidence",
        "literature", "investigate", "review",
    ),
    "daily": (
        "inbox", "email", "calendar", "schedule", "reminder", "todo",
        "follow up", "follow-up", "triage", "quick follow-up", "brief",
    ),
    "finance": (
        "finance", "financial", "budget", "expense", "expenses", "invoice",
        "transaction", "transactions", "bank", "tax", "taxes", "payroll",
        "ledger", "cashflow",
    ),
}

_DELEGATE_ONLY_PROFILES = frozenset({"coding"})


def build_profile_routing_candidates(multiplex: bool = True) -> list[ProfileRoutingCandidate]:
    """Return profile candidates with their descriptions attached."""
    candidates: list[ProfileRoutingCandidate] = []
    for name, home in profiles_to_serve(multiplex=multiplex):
        try:
            meta = read_profile_meta(home)
        except Exception:
            meta = {"description": "", "description_auto": False}
        candidates.append(
            ProfileRoutingCandidate(
                name=name,
                description=str(meta.get("description") or "").strip(),
                is_default=name == "default",
            )
        )
    return candidates


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _phrase_hits(text: str, phrases: Sequence[str]) -> int:
    hits = 0
    for phrase in phrases:
        if phrase in text:
            hits += 1
    return hits


def _score_candidate(candidate: ProfileRoutingCandidate, normalized_text: str, tokens: set[str]) -> int:
    if candidate.is_default or candidate.name == "default":
        return 0
    if candidate.name in _DELEGATE_ONLY_PROFILES:
        return 0

    score = 0
    name = candidate.name.lower()
    desc = candidate.description.lower()
    desc_tokens = _token_set(desc)

    # Generic description overlap helps custom profiles without hardcoded names.
    score += min(len(tokens & desc_tokens), 4)

    if name in normalized_text:
        score += 2

    score += _phrase_hits(normalized_text, _PROFILE_HINTS.get(candidate.name, ())) * 3

    # Lightly reward words that appear in the profile description itself.
    # This helps custom profiles when the user uses the same role language.
    score += _phrase_hits(normalized_text, tuple(sorted(desc_tokens & tokens))) // 2

    return score


def select_profile_for_message(
    message_text: str | None,
    candidates: Iterable[ProfileRoutingCandidate],
    *,
    current_profile: str = "default",
) -> str:
    """Pick the most appropriate profile for a message.

    Only the default profile is eligible for rerouting. Named profiles keep
    their current home so explicit selections and profile-specific gateways
    remain stable.
    """
    current = (current_profile or "").strip() or "default"
    if current != "default":
        return current

    normalized = _normalize_text(message_text)
    if not normalized or normalized.startswith("/"):
        return current

    tokens = _token_set(normalized)
    best_name = current
    best_score = 0
    second_score = 0

    for candidate in candidates:
        score = _score_candidate(candidate, normalized, tokens)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_name = candidate.name
        elif score > second_score:
            second_score = score

    if best_name == current:
        return current

    # Require a clear specialist signal. This keeps ordinary memory/file/chat
    # requests on the default profile instead of bouncing them around.
    if best_score < 4:
        return current
    if best_score - second_score < 1:
        return current
    return best_name


# ── Conversation stickiness ─────────────────────────────────────────────────
#
# The router scores each message independently, so without stickiness a
# follow-up like "thanks, now what?" after "debug this test" would fall back
# to the default profile and lose the specialist conversation. A pin keeps
# the conversation with the routed profile until it goes idle.

_STICKY_TTL_SECONDS = 900.0
_STICKY_MAX_ENTRIES = 512


def conversation_key(source: Any) -> Optional[tuple]:
    """Stable identity for one conversation, mirroring session grouping.

    Returns None when the source has no usable chat identity (stickiness is
    then skipped and per-message routing applies unchanged).
    """
    if source is None:
        return None
    chat_id = getattr(source, "chat_id", None)
    if chat_id is None or str(chat_id) == "":
        return None
    platform = getattr(source, "platform", None)
    return (
        str(getattr(platform, "value", platform)),
        str(chat_id),
        str(getattr(source, "thread_id", "") or ""),
        str(getattr(source, "user_id", "") or ""),
    )


class ProfileStickiness:
    """Remembers recent reroutes so follow-ups stay with the specialist.

    Pins expire after ``ttl_seconds`` of conversation inactivity (each use
    refreshes the clock). A fresh strong signal for a different profile
    re-pins immediately, and slash commands bypass routing entirely upstream,
    so ``/profile`` style overrides are never shadowed by a pin.
    """

    def __init__(
        self,
        ttl_seconds: float = _STICKY_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._pins: dict[tuple, tuple[str, float]] = {}

    def get(self, key: tuple) -> Optional[str]:
        entry = self._pins.get(key)
        if entry is None:
            return None
        profile, expires_at = entry
        if self._clock() >= expires_at:
            self._pins.pop(key, None)
            return None
        return profile

    def pin(self, key: tuple, profile: str) -> None:
        if len(self._pins) >= _STICKY_MAX_ENTRIES:
            self._evict()
        self._pins[key] = (profile, self._clock() + self._ttl)

    def refresh(self, key: tuple) -> None:
        entry = self._pins.get(key)
        if entry is not None:
            self._pins[key] = (entry[0], self._clock() + self._ttl)

    def clear(self, key: tuple) -> None:
        self._pins.pop(key, None)

    def _evict(self) -> None:
        now = self._clock()
        for k in [k for k, (_, exp) in self._pins.items() if now >= exp]:
            self._pins.pop(k, None)
        overflow = len(self._pins) - _STICKY_MAX_ENTRIES + 1
        if overflow > 0:
            for k, _ in sorted(self._pins.items(), key=lambda kv: kv[1][1])[:overflow]:
                self._pins.pop(k, None)
