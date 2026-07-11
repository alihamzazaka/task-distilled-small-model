"""Local inference wrapper with constrained-output enforcement.

The distilled student occasionally slips on format (a stray prose preamble,
a trailing comma, a missing field). Downstream systems must never crash on
that. This module wraps generation in a **generate -> parse -> validate ->
retry** loop so the caller always gets either a schema-valid, canonical
invoice dict or a *structured* error object it can route on.

Pipeline per request (SPEC §10 "format brittleness", architecture §"deployment"):

    1. generate      — student produces text for the fixed extraction prompt
    2. parse         — robustly pull the first JSON value out of that text
    3. validate      — coerce + validate against the pydantic Invoice schema
    4. retry         — on failure, re-ask with a targeted repair hint and a
                       little sampling temperature, up to `max_retries` times
    5. give up       — return ExtractionResult(data=None, error=...) — never
                       raise into the caller's hot path (use extract_safe)

Backends are pluggable:
- `TransformersBackend` — the real HF student (lazy-imports torch/transformers
  so this file imports fine on a laptop with no GPU stack installed).
- `FunctionBackend` — wraps any `callable(messages, temperature) -> str`; used
  by the `--demo` self-check and by tests, so the retry logic is exercisable
  with zero heavy dependencies.

The eval harness (`scripts/evaluate.py`) consumes `StructuredExtractor` and
`TransformersBackend` from here via the `serve/` path shim.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# Make `distil_task` importable whether infer.py is imported through the
# serve/ path shim or run directly (`python serve/infer.py`).
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from distil_task.io_utils import extract_first_json  # noqa: E402
from distil_task.prompts import (  # noqa: E402
    STUDENT_SYSTEM_PROMPT,
    build_extraction_user_prompt,
)
from distil_task.schema import Invoice, try_validate_invoice  # noqa: E402

Message = dict[str, str]


# ---------------------------------------------------------------------------
# Result / error types
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Outcome of one constrained extraction.

    `data` is a schema-valid, canonical invoice dict on success, or None when
    every attempt failed. `raw_text` is always the last raw model output (kept
    for failure analysis). `error` carries a readable reason on failure.
    """

    data: Optional[dict[str, Any]]
    raw_text: str
    valid: bool
    attempts: int
    error: Optional[str] = None
    invoice: Optional[Invoice] = None

    def to_dict(self) -> dict[str, Any]:
        """Structured, JSON-serializable view (the API/CLI response shape)."""
        return {
            "ok": self.valid,
            "data": self.data,
            "error": self.error,
            "attempts": self.attempts,
            "raw_text": self.raw_text,
        }


class ExtractionError(RuntimeError):
    """Raised by `StructuredExtractor.extract` when no attempt validated."""

    def __init__(self, message: str, result: ExtractionResult) -> None:
        super().__init__(message)
        self.result = result


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class Backend:
    """A text generator. Implementations turn a chat message list into text."""

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError


class FunctionBackend(Backend):
    """Wrap a plain `callable(messages, temperature) -> str`.

    Dependency-free — used for the `--demo` self-check and unit tests so the
    parse/validate/retry loop runs without torch or a real checkpoint.
    """

    def __init__(self, fn: Callable[[Sequence[Message], float], str]) -> None:
        self._fn = fn

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        return self._fn(messages, temperature)


class TransformersBackend(Backend):
    """The real student: a Hugging Face causal LM loaded from `model_dir`.

    torch/transformers are imported lazily on first generation so importing
    this module (and compiling it) needs no GPU stack. Model + tokenizer are
    loaded once and reused.
    """

    def __init__(
        self,
        model_dir: Path | str,
        max_new_tokens: int = 1024,
        device: Optional[str] = None,
        dtype: str = "auto",
    ) -> None:
        self.model_dir = str(model_dir)
        self.default_max_new_tokens = int(max_new_tokens)
        self.device = device
        self.dtype = dtype
        self._model: Any = None
        self._tok: Any = None
        self._torch: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415  (lazy: keep module importable on CPU-only laptop)
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        tok = AutoTokenizer.from_pretrained(self.model_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        if self.dtype == "auto":
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, self.dtype)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch_dtype,
            device_map="auto" if (self.device is None and torch.cuda.is_available()) else None,
        )
        if self.device is not None:
            model = model.to(self.device)
        model.eval()

        self._torch = torch
        self._tok = tok
        self._model = model

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        self._ensure_loaded()
        torch = self._torch
        templated = self._tok.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # transformers >=5 returns a BatchEncoding (dict) here, older returns a
        # bare tensor — support both.
        if hasattr(templated, "input_ids"):
            prompt_ids = templated["input_ids"]
        elif isinstance(templated, dict):
            prompt_ids = templated["input_ids"]
        else:
            prompt_ids = templated
        prompt_ids = prompt_ids.to(self._model.device)

        do_sample = bool(temperature and temperature > 0.0)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens or self.default_max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self._tok.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        with torch.no_grad():
            out = self._model.generate(prompt_ids, **gen_kwargs)
        new_tokens = out[0][prompt_ids.shape[-1]:]
        return self._tok.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Structured extractor (the constrained-output retry loop)
# ---------------------------------------------------------------------------

def _repair_hint(error: str) -> str:
    """The corrective user turn appended after an invalid attempt."""
    return (
        "Your previous reply was not accepted: "
        f"{error}\n"
        "Return ONLY a single JSON object that exactly matches the required "
        "schema — no prose, no markdown fences, no trailing text. All money "
        "fields must be plain numbers, the date must be ISO-8601 (YYYY-MM-DD), "
        "currency an ISO-4217 code, and every required key must be present."
    )


class StructuredExtractor:
    """Enforces schema-valid output from a `Backend` via bounded retries.

    `max_retries` is the number of *additional* attempts after the first, so
    the model is queried at most `max_retries + 1` times per document.
    """

    def __init__(
        self,
        backend: Backend,
        max_retries: int = 3,
        max_new_tokens: Optional[int] = None,
        retry_temperature: float = 0.3,
        money_abs_tol: float = 0.01,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.backend = backend
        self.max_retries = int(max_retries)
        self.max_new_tokens = max_new_tokens
        self.retry_temperature = float(retry_temperature)
        self.money_abs_tol = float(money_abs_tol)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _base_messages(document_text: str) -> list[Message]:
        return [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": build_extraction_user_prompt(document_text)},
        ]

    @staticmethod
    def _parse_and_validate(raw: str) -> tuple[Optional[Invoice], Optional[str]]:
        try:
            obj = extract_first_json(raw)
        except ValueError as e:
            return None, f"parse: {e}"
        inv, err = try_validate_invoice(obj)
        if inv is None:
            return None, f"schema: {err}"
        return inv, None

    # -- public API ---------------------------------------------------------

    def extract_safe(self, document_text: str) -> ExtractionResult:
        """Non-raising extraction. Always returns an ExtractionResult; on total
        failure `.data is None` and `.error` explains why."""
        base = self._base_messages(document_text)
        messages = list(base)
        last_raw = ""
        last_err: Optional[str] = "no attempts made"

        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            # first pass deterministic; retries add a little temperature to
            # escape a repeated bad generation.
            temperature = 0.0 if attempt == 1 else self.retry_temperature
            raw = self.backend.generate(
                messages,
                temperature=temperature,
                max_new_tokens=self.max_new_tokens,
            )
            last_raw = raw
            inv, err = self._parse_and_validate(raw)
            if inv is not None:
                return ExtractionResult(
                    data=inv.model_dump(mode="json"),
                    raw_text=raw,
                    valid=True,
                    attempts=attempt,
                    invoice=inv,
                )
            last_err = err
            # Re-ask with the failed reply + a targeted repair instruction.
            messages = base + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _repair_hint(err or "invalid output")},
            ]

        return ExtractionResult(
            data=None,
            raw_text=last_raw,
            valid=False,
            attempts=self.max_retries + 1,
            error=last_err,
        )

    def extract(self, document_text: str) -> dict[str, Any]:
        """Raising variant: returns the validated dict or raises
        ExtractionError (carrying the ExtractionResult for inspection)."""
        result = self.extract_safe(document_text)
        if result.data is None:
            raise ExtractionError(
                f"failed to produce schema-valid output after {result.attempts} "
                f"attempt(s): {result.error}",
                result,
            )
        return result.data


# ---------------------------------------------------------------------------
# CLI / self-check
# ---------------------------------------------------------------------------

_DEMO_DOCUMENT = """\
BRIGHTLEAF COFFEE ROASTERS
221 Market St, Seattle WA
Receipt #4471    2024-06-09

2x Espresso Blend 250g   14.00   28.00
1x Ceramic Mug            12.50   12.50

Subtotal                          40.50
Sales Tax 10%                      4.05
TOTAL USD                         44.55
"""


def _demo_backend() -> FunctionBackend:
    """A scripted student: the FIRST reply is deliberately malformed (prose +
    a money field as a string), the SECOND is valid. Proves the retry loop
    recovers a schema slip without any GPU/model."""
    state = {"calls": 0}

    def fn(messages: Sequence[Message], temperature: float) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            return (
                "Sure! Here is the invoice data:\n"
                '{"vendor": "Brightleaf Coffee Roasters", "date": "06/09/2024", '
                '"currency": "$", "line_items": [], "subtotal": "40.50", '
                '"tax": "4.05", "grand_total": "44.55"}'
            )
        return (
            '{"vendor": "Brightleaf Coffee Roasters", "date": "2024-06-09", '
            '"currency": "USD", "line_items": ['
            '{"description": "Espresso Blend 250g", "qty": 2, "unit_price": 14.00, "total": 28.00}, '
            '{"description": "Ceramic Mug", "qty": 1, "unit_price": 12.50, "total": 12.50}], '
            '"subtotal": 40.50, "tax": 4.05, "grand_total": 44.55, "payment_terms": null}'
        )

    return FunctionBackend(fn)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Constrained invoice extraction.")
    ap.add_argument("--model", default=None, help="student checkpoint dir (real inference)")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--demo", action="store_true",
                    help="run the dependency-free scripted self-check (no GPU)")
    ap.add_argument("--file", default=None, help="read the document text from a file")
    args = ap.parse_args(argv)

    if args.demo or args.model is None:
        backend: Backend = _demo_backend()
        document = _DEMO_DOCUMENT
        if not args.demo:
            print("[infer] no --model given; running the scripted --demo backend.\n")
    else:
        backend = TransformersBackend(args.model, max_new_tokens=args.max_new_tokens)
        document = Path(args.file).read_text(encoding="utf-8") if args.file else _DEMO_DOCUMENT

    extractor = StructuredExtractor(
        backend, max_retries=args.max_retries, max_new_tokens=args.max_new_tokens
    )
    result = extractor.extract_safe(document)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
