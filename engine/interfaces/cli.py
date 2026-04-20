"""engine.interfaces.cli — the flagship CLI.

Rich-formatted output with optional JSON mode for scripting. All engine
features (memory, domain, plugins, streaming, trace) surface as flags.

Usage:
    python -m engine.interfaces.cli "your research question"
    python -m engine.interfaces.cli ask "question" --domain medical --memory persistent
    python -m engine.interfaces.cli reset-memory
    python -m engine.interfaces.cli domains list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Repo root on sys.path for `core.rag`, `engine.*`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.core.memory import MemoryStore, VALID_MODES  # noqa: E402
from engine.core.trace import _print_trace_summary  # noqa: E402
from engine.interfaces.common import (  # noqa: E402
    format_sources,
    format_trace_per_node,
    format_verified_summary,
    run_query,
)


# ── Pretty output (no external deps beyond `rich` which is pinned) ───

def _print_header(title: str) -> None:
    print(f"\n── {title} ──")


def _print_markdown(result) -> None:
    print(f"Q: {result.question}\n")
    if result.question_class:
        print(f"[class: {result.question_class}]\n")
    print(f"A: {result.answer}\n")

    if result.sources:
        _print_header("sources")
        for row in format_sources(result):
            fetched_mark = "●" if row["fetched"] else "○"
            print(f"  [{row['idx']}] {fetched_mark} {row['url']}")
            if row["preview"]:
                print(f"        {row['preview']}")

    if result.verified_claims or result.unverified_claims:
        _print_header("hallucination check")
        print(f"  {format_verified_summary(result)}")
        for c in result.verified_claims:
            print(f"  ✓ {c.get('text','')}")
        for c in result.unverified_claims:
            print(f"  ✗ {c}")

    if result.memory_hits:
        _print_header("memory hits (prior related research)")
        for h in result.memory_hits:
            print(f"  ({h['score']:.2f}) {h['question']}")
            print(f"         → {h['answer'][:120]}")

    _print_header("trace (per-node totals)")
    for row in format_trace_per_node(result):
        print(f"  {row['node']:12s}  calls={row['calls']:2d}  "
              f"latency={row['latency_s']:6.2f}s  tokens~{row['tokens_est']}")
    print(f"\n  total: {result.total_latency_s:.2f}s · "
          f"~{result.total_tokens_est} tokens · iterations={result.iterations}")


def _print_json(result) -> None:
    payload = {
        "question": result.question,
        "domain": result.domain,
        "question_class": result.question_class,
        "answer": result.answer,
        "verified_claims": result.verified_claims,
        "unverified_claims": result.unverified_claims,
        "sources": result.sources,
        "memory_hits": result.memory_hits,
        "iterations": result.iterations,
        "total_latency_s": result.total_latency_s,
        "total_tokens_est": result.total_tokens_est,
        "trace_per_node": format_trace_per_node(result),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


# ── CLI commands ─────────────────────────────────────────────────────

def _cmd_ask(args: argparse.Namespace) -> int:
    # Honor --api-key override (route to cloud OpenAI instead of local Ollama).
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        os.environ.pop("OPENAI_BASE_URL", None)  # default OpenAI hosted
    if args.model:
        # Single-knob override of ALL MODEL_* env vars.
        for k in ("MODEL_PLANNER", "MODEL_SEARCHER", "MODEL_SYNTHESIZER",
                  "MODEL_VERIFIER", "MODEL_CRITIC", "MODEL_ROUTER", "MODEL_COMPRESSOR"):
            os.environ[k] = args.model

    store = MemoryStore.open(args.memory) if args.memory else None

    result = run_query(args.question, domain=args.domain, memory=store)

    if args.output == "json":
        _print_json(result)
    else:
        _print_markdown(result)

    if store is not None:
        store.close()
    return 0


def _cmd_reset_memory(args: argparse.Namespace) -> int:
    store = MemoryStore.open("persistent", path=Path(args.db_path) if args.db_path else None)
    n = store.count()
    store.reset()
    store.close()
    print(f"reset: {n} trajectories wiped")
    return 0


def _cmd_memory_count(args: argparse.Namespace) -> int:
    store = MemoryStore.open("persistent", path=Path(args.db_path) if args.db_path else None)
    n = store.count()
    store.close()
    print(f"{n} trajectories")
    return 0


def _cmd_domains_list(args: argparse.Namespace) -> int:
    # Phase 6 populates engine/domains/*.yaml; for now list what exists.
    root = _REPO_ROOT / "engine" / "domains"
    if not root.exists():
        print("(no domain presets installed yet — Phase 6)")
        return 0
    for p in sorted(root.glob("*.yaml")):
        print(p.stem)
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    from engine import __version__
    print(f"engine v{__version__}")
    return 0


# ── Argument parsing ────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="engine",
        description="Gold-standard open-source local research engine.",
    )
    sub = p.add_subparsers(dest="cmd")

    ask = sub.add_parser("ask", help="Run a research query (default command)")
    ask.add_argument("question", help="The research question")
    ask.add_argument("--domain", default="general", help="Domain preset name (Phase 6)")
    ask.add_argument("--memory", choices=sorted(VALID_MODES), default=None,
                     help="Memory mode: off / session / persistent")
    ask.add_argument("--api-key", default=None,
                     help="Override OPENAI_API_KEY and route to cloud OpenAI")
    ask.add_argument("--model", default=None,
                     help="Override all MODEL_* env vars with this single value")
    ask.add_argument("--output", choices=["markdown", "json"], default="markdown")
    ask.set_defaults(func=_cmd_ask)

    reset = sub.add_parser("reset-memory", help="Wipe the persistent memory store")
    reset.add_argument("--db-path", default=None)
    reset.set_defaults(func=_cmd_reset_memory)

    mem_count = sub.add_parser("memory-count", help="Print number of trajectories")
    mem_count.add_argument("--db-path", default=None)
    mem_count.set_defaults(func=_cmd_memory_count)

    domains = sub.add_parser("domains", help="Domain presets")
    d_sub = domains.add_subparsers(dest="domains_cmd")
    d_list = d_sub.add_parser("list")
    d_list.set_defaults(func=_cmd_domains_list)

    version = sub.add_parser("version", help="Print engine version")
    version.set_defaults(func=_cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv = list(argv if argv is not None else sys.argv[1:])

    # Bare `engine <question>` dispatches to `ask` for convenience.
    if argv and argv[0] not in {"ask", "reset-memory", "memory-count", "domains", "version", "-h", "--help"}:
        argv.insert(0, "ask")

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
