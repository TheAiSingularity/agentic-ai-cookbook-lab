"""engine.core.models — LLM plumbing, model routing, small-model heuristic.

All OpenAI-compatible calls (batched + streaming) go through `_chat` /
`_chat_stream`. Both append a trace entry (via `engine.core.trace._TRACE_BUFFER`)
when `ENABLE_TRACE=1`. Streaming falls back to batched on backend error.

Model name env vars default to cloud names (gpt-5-mini / gpt-5-nano) but the
entire stack honors `OPENAI_BASE_URL` so Ollama / vLLM / SGLang drop in with
one env flip. The small-model heuristic auto-shrinks TOP_K_EVIDENCE when a
4 B-class local model is detected (regex matches `:e2b`, `:1-4b`, `-1-4b`,
`nano` but NOT `mini` — cloud "mini" models handle context fine).
"""

from __future__ import annotations

import os
import re
import sys
import time

from openai import OpenAI

from engine.core import trace as _trace

ENV = os.environ.get

# Model routing
MODEL_PLANNER = ENV("MODEL_PLANNER", "gpt-5-nano")
MODEL_SEARCHER = ENV("MODEL_SEARCHER", "gpt-5-mini")
MODEL_SYNTHESIZER = ENV("MODEL_SYNTHESIZER", "gpt-5-mini")
MODEL_VERIFIER = ENV("MODEL_VERIFIER", MODEL_PLANNER)
MODEL_CRITIC = ENV("MODEL_CRITIC", MODEL_PLANNER)
MODEL_ROUTER = ENV("MODEL_ROUTER", MODEL_PLANNER)
MODEL_COMPRESSOR = ENV("MODEL_COMPRESSOR", MODEL_PLANNER)

# W7 streaming toggle
ENABLE_STREAM = ENV("ENABLE_STREAM", "1") == "1"

# W6.3 — small-model regex. Matches `:e2b`, `:1-4b`, `-1-4b`, `nano`. Does NOT
# match `mini` (gpt-5-mini / gpt-4o-mini are cloud-hosted and capable).
_SMALL_MODEL_RE = re.compile(r"(:[e]?[1-4]b\b|[-_][1-4]b\b|\bnano\b)", re.IGNORECASE)


def _default_top_k(model_synth: str, explicit: str | None) -> int:
    """Return TOP_K_EVIDENCE, auto-shrunk for small models unless overridden."""
    if explicit is not None:
        return int(explicit)
    if _SMALL_MODEL_RE.search(model_synth or ""):
        return int(ENV("SMALL_MODEL_TOPK", "5"))
    return 8


def _llm() -> OpenAI:
    """Build an OpenAI client honoring OPENAI_BASE_URL (Ollama / vLLM / cloud)."""
    return OpenAI(api_key=ENV("OPENAI_API_KEY", "ollama"), base_url=ENV("OPENAI_BASE_URL"))


def _supports_temperature(model: str) -> bool:
    """GPT-5 family rejects any non-default temperature (must omit the kwarg)."""
    return not (model or "").lower().startswith("gpt-5")


def _chat(model: str, prompt: str, temperature: float = 0.0) -> str:
    """Batched chat completion. Appends a trace entry when ENABLE_TRACE=1."""
    t0 = time.monotonic()
    kwargs: dict = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if _supports_temperature(model):
        kwargs["temperature"] = temperature
    resp = _llm().chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    if _trace.ENABLE_TRACE:
        _trace._TRACE_BUFFER.append({
            "model": model,
            "latency_s": round(time.monotonic() - t0, 3),
            "prompt_chars": len(prompt),
            "response_chars": len(content),
            "tokens_est": (len(prompt) + len(content)) // 4,
        })
    return content


def _chat_stream(model: str, prompt: str, temperature: float = 0.0, sink=None) -> str:
    """W7 — stream tokens to `sink` while accumulating the full response.

    Falls back to `_chat` if the backend rejects `stream=True`. Trace entry
    matches `_chat` plus a `streamed: True` marker.
    """
    if sink is None:
        def sink(tok: str) -> None:
            sys.stdout.write(tok)
            sys.stdout.flush()

    t0 = time.monotonic()
    stream_kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if _supports_temperature(model):
        stream_kwargs["temperature"] = temperature
    try:
        stream = _llm().chat.completions.create(**stream_kwargs)
    except Exception as exc:  # noqa: BLE001 — backend may reject streaming; fall back
        # Only print on stderr when the user has trace enabled — otherwise
        # silent fallback keeps scripted runs clean.
        if _trace.ENABLE_TRACE:
            print(f"[stream] backend rejected streaming, falling back to batched: "
                  f"{type(exc).__name__}", file=sys.stderr)
        return _chat(model, prompt, temperature)

    pieces: list[str] = []
    try:
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            tok = (getattr(delta, "content", None) or "") if delta else ""
            if tok:
                pieces.append(tok)
                sink(tok)
    except Exception as exc:  # noqa: BLE001 — mid-stream break shouldn't crash the pipeline
        if _trace.ENABLE_TRACE:
            print(f"[stream] mid-stream error after {len(pieces)} tokens: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
    sink("\n")

    content = "".join(pieces)
    if _trace.ENABLE_TRACE:
        _trace._TRACE_BUFFER.append({
            "model": model,
            "latency_s": round(time.monotonic() - t0, 3),
            "prompt_chars": len(prompt),
            "response_chars": len(content),
            "tokens_est": (len(prompt) + len(content)) // 4,
            "streamed": True,
        })
    return content


__all__ = [
    "MODEL_PLANNER", "MODEL_SEARCHER", "MODEL_SYNTHESIZER", "MODEL_VERIFIER",
    "MODEL_CRITIC", "MODEL_ROUTER", "MODEL_COMPRESSOR",
    "ENABLE_STREAM", "_SMALL_MODEL_RE", "_default_top_k",
    "_llm", "_chat", "_chat_stream", "_supports_temperature", "OpenAI",
]
