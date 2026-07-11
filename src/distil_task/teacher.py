"""Provider-pluggable teacher client.

- Default provider: Anthropic (reads ANTHROPIC_API_KEY from the env).
- Disk response cache in data/cache/ keyed by a hash of
  (provider, model, system, prompt, temperature, max_tokens) so re-runs
  are free and deterministic-at-temperature-0 passes never re-bill.
- Retry with exponential backoff + jitter on transient failures.
- Per-call cost accounting from the configurable price table in
  configs/default.yaml (USD per million tokens).

To add a provider: subclass TeacherClient, implement `_complete`, and
register it in `get_teacher`.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import ensure_dir


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    """Accumulates token usage and dollars across calls (cache hits are $0)."""

    price_table: dict[str, dict[str, float]] = field(default_factory=dict)
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    def price_for(self, model: str) -> tuple[float, float]:
        row = self.price_table.get(model)
        if row is None:
            # prefix match so "claude-sonnet-4-5-20250929" hits "claude-sonnet-4-5"
            for key, r in self.price_table.items():
                if model.startswith(key) or key.startswith(model):
                    row = r
                    break
        if row is None:
            return (0.0, 0.0)
        return (float(row.get("input_per_mtok", 0.0)), float(row.get("output_per_mtok", 0.0)))

    def record(self, model: str, input_tokens: int, output_tokens: int, cached: bool) -> float:
        if cached:
            self.cache_hits += 1
            return 0.0
        pin, pout = self.price_for(model)
        cost = input_tokens / 1e6 * pin + output_tokens / 1e6 * pout
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.usd += cost
        return cost

    def report(self) -> dict[str, Any]:
        n = max(self.calls, 1)
        return {
            "billed_calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "avg_input_tokens_per_call": round(self.input_tokens / n, 1),
            "avg_output_tokens_per_call": round(self.output_tokens / n, 1),
            "total_usd": round(self.usd, 4),
            "usd_per_1k_calls": round(self.usd / n * 1000, 4),
        }


# ---------------------------------------------------------------------------
# Base client with cache + retry
# ---------------------------------------------------------------------------

@dataclass
class TeacherResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cached: bool
    cost_usd: float


class TeacherError(RuntimeError):
    pass


class TeacherClient:
    """Base class: caching, retry/backoff, cost accounting. Subclasses
    implement `_complete(system, prompt, temperature, max_tokens)`."""

    provider = "base"

    def __init__(
        self,
        model: str,
        cache_dir: Path | str,
        price_table: Optional[dict[str, dict[str, float]]] = None,
        max_retries: int = 5,
        backoff_base_s: float = 2.0,
    ) -> None:
        self.model = model
        self.cache_dir = ensure_dir(Path(cache_dir))
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.cost = CostTracker(price_table=price_table or {})

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache_salt: str = "",
        use_cache: bool = True,
    ) -> TeacherResponse:
        key = self._cache_key(prompt, system, temperature, max_tokens, cache_salt)
        cache_path = self.cache_dir / f"{key}.json"
        if use_cache and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.cost.record(self.model, 0, 0, cached=True)
            return TeacherResponse(
                text=payload["text"],
                input_tokens=payload.get("input_tokens", 0),
                output_tokens=payload.get("output_tokens", 0),
                cached=True,
                cost_usd=0.0,
            )

        text, in_tok, out_tok = self._complete_with_retry(system, prompt, temperature, max_tokens)
        cost = self.cost.record(self.model, in_tok, out_tok, cached=False)
        if use_cache:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "provider": self.provider,
                        "model": self.model,
                        "temperature": temperature,
                        "cache_salt": cache_salt,
                        "text": text,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            tmp.replace(cache_path)
        return TeacherResponse(text=text, input_tokens=in_tok, output_tokens=out_tok, cached=False, cost_usd=cost)

    # -- internals ----------------------------------------------------------

    def _cache_key(self, prompt: str, system: str, temperature: float, max_tokens: int, salt: str) -> str:
        blob = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "salt": salt,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:40]

    def _complete_with_retry(
        self, system: str, prompt: str, temperature: float, max_tokens: int
    ) -> tuple[str, int, int]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._complete(system, prompt, temperature, max_tokens)
            except Exception as e:  # provider SDK errors vary; classify below
                last_err = e
                if not self._is_retryable(e) or attempt == self.max_retries:
                    break
                delay = self.backoff_base_s * (2**attempt) + random.uniform(0, 1)
                time.sleep(min(delay, 60.0))
        raise TeacherError(
            f"{self.provider}:{self.model} failed after {self.max_retries + 1} attempts: {last_err}"
        ) from last_err

    @staticmethod
    def _is_retryable(e: Exception) -> bool:
        name = type(e).__name__.lower()
        msg = str(e).lower()
        retryable_markers = (
            "ratelimit", "rate_limit", "overloaded", "timeout", "timed out",
            "connection", "apiconnection", "internalserver", "internal server",
            "529", "500", "502", "503", "504",
        )
        return any(m in name or m in msg for m in retryable_markers)

    def _complete(self, system: str, prompt: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Anthropic provider (default)
# ---------------------------------------------------------------------------

class AnthropicTeacher(TeacherClient):
    provider = "anthropic"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client = None  # lazy: module stays importable without the SDK

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415 (lazy on purpose)
            except ImportError as e:
                raise TeacherError(
                    "The 'anthropic' package is not installed. "
                    "pip install -r requirements-laptop.txt"
                ) from e
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise TeacherError(
                    "ANTHROPIC_API_KEY is not set in the environment."
                )
            self._client = anthropic.Anthropic()
        return self._client

    def _complete(self, system: str, prompt: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        return text, in_tok, out_tok


# ---------------------------------------------------------------------------
# Local open-model provider (OpenAI-compatible: Ollama / llama.cpp / vLLM)
# ---------------------------------------------------------------------------

class LocalOpenAITeacher(TeacherClient):
    """Teacher backed by a local Ollama server.

    Reads OLLAMA_BASE_URL (default http://localhost:11434) and posts to the
    native /api/chat endpoint with ``think: false`` so reasoning models
    (e.g. qwen3:14b) emit the answer directly instead of a <think> block —
    faster, cheaper, and it keeps the output clean for JSON extraction. Uses
    only the stdlib (urllib); no SDK required. JSON is obtained via the
    extraction prompt, not a forced format, because the same client also drives
    free-text seed generation (scripts/00).
    """

    provider = "local_openai"

    def _complete(self, system: str, prompt: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
        import urllib.request  # noqa: PLC0415 (stdlib, lazy)

        base = os.environ.get(
            "OLLAMA_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:11434").replace("/v1", ""),
        ).rstrip("/")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "think": False,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/chat", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 (local endpoint)
            payload = json.loads(resp.read().decode("utf-8"))
        text = (payload.get("message") or {}).get("content", "") or ""
        in_tok = int(payload.get("prompt_eval_count", 0) or 0)
        out_tok = int(payload.get("eval_count", 0) or 0)
        return text, in_tok, out_tok


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[TeacherClient]] = {
    "anthropic": AnthropicTeacher,
    "local_openai": LocalOpenAITeacher,
}


def register_provider(name: str, cls: type[TeacherClient]) -> None:
    _PROVIDERS[name.lower()] = cls


def get_teacher(cfg: dict[str, Any], cache_dir: Path | str) -> TeacherClient:
    """Build the teacher client from the `teacher:` section of the config."""
    tcfg = cfg["teacher"]
    provider = str(tcfg.get("provider", "anthropic")).lower()
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise TeacherError(
            f"Unknown teacher provider {provider!r}. Registered: {sorted(_PROVIDERS)}"
        )
    return cls(
        model=tcfg["model"],
        cache_dir=cache_dir,
        price_table=tcfg.get("price_table", {}),
        max_retries=int(tcfg.get("max_retries", 5)),
        backoff_base_s=float(tcfg.get("backoff_base_s", 2.0)),
    )
