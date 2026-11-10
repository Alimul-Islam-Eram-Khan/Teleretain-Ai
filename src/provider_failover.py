"""Fail-closed, free-tier-acknowledged Gemini -> Groq LLM failover.

No billing API can guarantee that an account is free. Users MUST confirm
that both projects are on free plans and have no paid billing enabled.
Only retryable provider errors trigger failover; unsafe LLM output does NOT.
This module never sees raw model input features.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import time
from pathlib import Path
from typing import Callable, TypeVar

ROOT = Path(__file__).resolve().parents[1]
GEMINI_MODELS = frozenset({"gemini-3.8-flash", "gemini-3.5-flash-lite"})
GROQ_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b"})
COOLDOWN_SECONDS = 120
_cooldown_until: dict[str, float] = {}


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    key: str  # secrets are never logged or returned to the UI


@dataclass(frozen=True)
class ProviderAnswer:
    text: str
    provider: str
    model: str


class ProvidersUnavailable(RuntimeError):
    """Neither provider can currently be used without paid billing."""


def _enabled(name: str) -> bool:
    return os.getenv(f"{name.upper()}_FREE_TIER_CONFIRMED", "").strip().lower() == "true"


def configured_providers() -> list[Provider]:
    """Only human-confirmed, allowlisted providers; NO automatic paid upgrades."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    ordered = os.getenv("COPILOT_PRIMARY", "gemini").strip().lower()
    if ordered not in {"gemini", "groq"}:
        raise ValueError("COPILOT_PRIMARY must be 'gemini' or 'groq'.")
    providers: dict[str, Provider] = {}
    specs = (
        ("gemini", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.8-flash", GEMINI_MODELS),
        ("groq", "GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-20b", GROQ_MODELS),
    )
    for name, key_var, model_var, default, allowlist in specs:
        if not _enabled(name):
            continue
        key = os.getenv(key_var, "").strip()
        if not key:
            continue
        model = os.getenv(model_var, default).strip() or default
        if model not in allowlist:
            # Fail closed. Never send a request to a model not reviewed for free tier.
            continue
        providers[name] = Provider(name, model, key)
    order = ("gemini", "groq") if ordered == "gemini" else ("groq", "gemini")
    return [providers[name] for name in order if name in providers]


def create_llm(spec: Provider):
    """LangChain-compatible provider; only invoke after Stage 5 validation."""
    if spec.name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=spec.model, api_key=spec.key, vertexai=False,
            temperature=0, reasoning_effort="low",
            max_output_tokens=1024, max_retries=0,
        )
    if spec.name == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=spec.model, api_key=spec.key, temperature=0,
            max_tokens=1024, max_retries=0, timeout=20,
        )
    raise ValueError("Unrecognized LLM provider")


def _retryable(exc: Exception) -> bool:
    """Only quota errors, server errors and transport failures can fail over.

    IMPORTANT: 400/401/403/refusal/safety violations never trigger a second call.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if isinstance(status, int):
        return status in (408, 429) or status >= 500
    # Provider SDKs may encode errors in a class name without status_code.
    known = {
        "ResourceExhausted", "RateLimitError", "TooManyRequests",
        "ServiceUnavailable", "InternalServerError", "BadGateway", "GatewayTimeout",
        "APIConnectionError", "APITimeoutError", "DeadlineExceeded",
        "ConnectError", "ReadTimeout", "ConnectTimeout", "TimeoutException",
    }
    if type(exc).__name__ in known:
        return True
    # Standard library timeout/connection failures (no text-based matching).
    return isinstance(exc, (TimeoutError, ConnectionError))


def _available(name: str) -> bool:
    return time.monotonic() >= _cooldown_until.get(name, 0)


def _mark_unavailable(name: str) -> None:
    _cooldown_until[name] = time.monotonic() + COOLDOWN_SECONDS


def clear_cooldowns_for_testing() -> None:
    """Test-only helper; does not reset any actual provider quota."""
    _cooldown_until.clear()


def run_with_failover(action: Callable[[Provider, object], str]) -> ProviderAnswer:
    """A provider's whole operation is retried with backup ONLY for transient errors.

    In-memory cooldown stops re-trying exhausted provider for 120 seconds in
    a running chat session, then probes it again; no quota bypass is attempted.
    """
    providers = configured_providers()
    if not providers:
        raise ProvidersUnavailable("No confirmed Free Tier LLM provider is configured.")
    last_kind = None
    for provider in providers:
        if not _available(provider.name):
            continue
        try:
            client = create_llm(provider)
            text = action(provider, client)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("No safe explanation returned; offline answer required.")
            return ProviderAnswer(text.strip(), provider.name, provider.model)
        except Exception as exc:
            if not _retryable(exc):
                # NEVER fall through for invalid input, auth, unsafe output, refusals.
                raise
            _mark_unavailable(provider.name)
            last_kind = type(exc).__name__
            continue
    raise ProvidersUnavailable(f"Confirmed Free Tier providers are temporarily unavailable ({last_kind or 'cooldown'}).")
