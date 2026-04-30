"""engine.core — the reusable research pipeline, trace, and model routing.

Three modules:
  - `trace`    — W4.3 observability (trace buffer, node-level merge, CLI printer)
  - `models`   — LLM plumbing (_chat, _chat_stream), model-name env vars,
                 small-model heuristic (_SMALL_MODEL_RE, _default_top_k).
  - `pipeline` — the 8-node LangGraph + all techniques (T2/T4/W4/W6/W7).

All public symbols are re-exported at this level so downstream code can
`from engine.core import build_graph` without caring about submodules.

Phase 2 adds `memory` and `compaction` modules alongside these three.
Phase 3's three interfaces (cli, tui, web) all import from here.
"""

from engine.core.models import (
    ENABLE_STREAM,
    MODEL_COMPRESSOR,
    MODEL_CRITIC,
    MODEL_PLANNER,
    MODEL_ROUTER,
    MODEL_SEARCHER,
    MODEL_SYNTHESIZER,
    MODEL_VERIFIER,
    _chat,
    _chat_stream,
    _default_top_k,
    _llm,
    _SMALL_MODEL_RE,
)
from engine.core.pipeline import (
    State,
    _after_verify,
    _classify,
    _compress,
    _corpus_hits,
    _critic,
    _fetch_one,
    _fetch_url,
    _flare_augment,
    _get_corpus,
    _get_reranker,
    _grounding_score,
    _hyde_expand,
    _plan,
    _retrieve,
    _exa,
    _search,
    _search_one,
    _searxng,
    _synthesize,
    _web_search,
    _synthesize_once,
    _verify,
    build_graph,
)
from engine.core.trace import (
    ENABLE_TRACE,
    _TRACE_BUFFER,
    _drain_trace,
    _merge_trace,
    _print_trace_summary,
)

__all__ = [
    # Graph entry point
    "build_graph",
    "State",
    # Models
    "MODEL_PLANNER", "MODEL_SEARCHER", "MODEL_SYNTHESIZER", "MODEL_VERIFIER",
    "MODEL_CRITIC", "MODEL_ROUTER", "MODEL_COMPRESSOR",
    "ENABLE_STREAM", "_SMALL_MODEL_RE", "_default_top_k",
    "_llm", "_chat", "_chat_stream",
    # Trace
    "ENABLE_TRACE", "_TRACE_BUFFER", "_drain_trace", "_merge_trace", "_print_trace_summary",
    # Nodes (exported for tests + direct callers)
    "_classify", "_plan", "_hyde_expand", "_search", "_search_one", "_retrieve",
    "_fetch_url", "_fetch_one", "_compress", "_synthesize", "_synthesize_once",
    "_flare_augment", "_verify", "_after_verify", "_critic", "_grounding_score",
    "_searxng", "_exa", "_web_search", "_get_corpus", "_corpus_hits", "_get_reranker",
]
