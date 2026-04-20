"""engine.interfaces.tui — Textual-based terminal UI.

Keyboard-driven research workbench. Works in any terminal (iTerm, native
Terminal.app, SSH session). No browser, no Electron.

Panes:
  - Top bar       query input + domain selector + memory toggle
  - Left panel    sources gallery (clickable rows)
  - Center panel  answer (streaming) + hallucination-check below
  - Right panel   trace timeline (per-node) + memory hits
  - Bottom bar    status / hints

Usage:
    python -m engine.interfaces.tui

Requires `textual>=0.80`. If not installed, prints a friendly error.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        ListItem,
        ListView,
        Label,
        Select,
        Static,
    )
except ImportError:  # pragma: no cover — handled at runtime
    print(
        "[engine.tui] Textual is not installed. Run `pip install -r engine/requirements.txt` "
        "(or just `pip install textual>=0.80`) and retry.",
        file=sys.stderr,
    )
    raise SystemExit(1)

from engine.core.memory import MemoryStore  # noqa: E402
from engine.interfaces.common import (  # noqa: E402
    format_sources,
    format_trace_per_node,
    format_verified_summary,
    run_query,
)


CSS = """
Screen { layout: vertical; background: $surface; }
#topbar { height: 5; padding: 1; background: $boost; }
#main   { height: 1fr; }
#status { height: 1; padding: 0 1; background: $primary-background; color: $text; }

#sources   { width: 35%; padding: 1; border: round $accent; }
#center    { width: 40%; padding: 1; border: round $primary; }
#right     { width: 25%; padding: 1; border: round $accent; }

.section-title { color: $accent; text-style: bold; margin-bottom: 1; }
.hit-verified { color: $success; }
.hit-unverified { color: $error; }
.trace-row  { color: $text-muted; }

Input { width: 100%; }
"""


class EngineTUI(App):
    TITLE = "engine — local research"
    CSS = CSS
    BINDINGS = [
        Binding("enter", "ask", "Ask", show=True),
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("ctrl+m", "toggle_memory", "Memory", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    memory_mode: reactive[str] = reactive("session")
    domain: reactive[str] = reactive("general")

    def __init__(self):
        super().__init__()
        self._store: MemoryStore | None = None

    # ── Compose layout ────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="topbar"):
            yield Input(placeholder="Ask a research question…", id="q")
            with Horizontal():
                yield Select(
                    [("general", "general"), ("medical", "medical"), ("papers", "papers"),
                     ("financial", "financial"), ("stock_trading", "stock_trading"),
                     ("personal_docs", "personal_docs")],
                    value="general", id="domain_sel", prompt="domain",
                )
                yield Select(
                    [("off", "off"), ("session", "session"), ("persistent", "persistent")],
                    value="session", id="memory_sel", prompt="memory",
                )
                yield Button("Ask", id="ask_btn", variant="primary")
                yield Button("Reset memory", id="reset_btn")

        with Horizontal(id="main"):
            with VerticalScroll(id="sources"):
                yield Static("[no run yet]", id="sources_body")

            with VerticalScroll(id="center"):
                yield Static("engine ready — enter a question above and press Enter.",
                             id="center_body")

            with VerticalScroll(id="right"):
                yield Static("[trace appears here]", id="right_body")

        yield Static("Ready.", id="status")
        yield Footer()

    # ── Memory ────────────────────────────────────────────────

    def _ensure_store(self) -> MemoryStore:
        if self._store is None or getattr(self._store, "_mode", None) != self.memory_mode:
            if self._store is not None:
                self._store.close()
            self._store = MemoryStore.open(self.memory_mode)
            self._store._mode = self.memory_mode  # type: ignore[attr-defined]
        return self._store

    # ── Actions ───────────────────────────────────────────────

    def action_ask(self) -> None:
        q = self.query_one("#q", Input).value.strip()
        if not q:
            return
        self._run_query(q)

    def action_clear(self) -> None:
        self.query_one("#sources_body", Static).update("[cleared]")
        self.query_one("#center_body", Static).update("engine ready.")
        self.query_one("#right_body", Static).update("[trace cleared]")
        self.query_one("#status", Static).update("Cleared.")

    def action_toggle_memory(self) -> None:
        order = ["off", "session", "persistent"]
        idx = order.index(self.memory_mode)
        self.memory_mode = order[(idx + 1) % len(order)]
        self.query_one("#memory_sel", Select).value = self.memory_mode
        self.query_one("#status", Static).update(f"Memory: {self.memory_mode}")

    # ── Event handlers ────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ask_btn":
            self.action_ask()
        elif event.button.id == "reset_btn":
            store = self._ensure_store()
            n = store.count()
            store.reset()
            self.query_one("#status", Static).update(f"Memory reset: {n} trajectories wiped.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "q":
            self._run_query(event.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "domain_sel":
            self.domain = event.value
        elif event.select.id == "memory_sel":
            self.memory_mode = event.value
            self.query_one("#status", Static).update(f"Memory: {self.memory_mode}")

    # ── Query execution ───────────────────────────────────────

    def _run_query(self, question: str) -> None:
        status = self.query_one("#status", Static)
        status.update(f"Running: {question[:60]}…")
        self.refresh()

        store = self._ensure_store() if self.memory_mode != "off" else None
        result = run_query(question, domain=self.domain, memory=store)

        # ── Sources pane ──
        src_body = self.query_one("#sources_body", Static)
        if not result.sources:
            src_body.update("[no sources retrieved]")
        else:
            lines = ["[b]Sources[/b]", ""]
            for row in format_sources(result, max_chars=100):
                fmark = "●" if row["fetched"] else "○"
                lines.append(f"[b]{fmark} [{row['idx']}] {row['title']}[/b]")
                lines.append(f"   [dim]{row['url']}[/dim]")
                if row["preview"]:
                    lines.append(f"   {row['preview']}")
                lines.append("")
            src_body.update("\n".join(lines))

        # ── Center pane: answer + hallucination check ──
        center_body = self.query_one("#center_body", Static)
        sections = [
            f"[b]Q:[/b] {result.question}",
            f"[dim]class: {result.question_class or '?'}[/dim]",
            "",
            f"[b]A:[/b] {result.answer}",
            "",
            f"[b]Hallucination check[/b] — {format_verified_summary(result)}",
        ]
        for c in result.verified_claims:
            sections.append(f"  [green]✓[/green] {c.get('text','')}")
        for c in result.unverified_claims:
            sections.append(f"  [red]✗[/red] {c}")
        center_body.update("\n".join(sections))

        # ── Right pane: trace + memory hits ──
        right_body = self.query_one("#right_body", Static)
        lines = ["[b]Trace (per node)[/b]", ""]
        for row in format_trace_per_node(result):
            lines.append(f"  {row['node']:10s} calls={row['calls']:2d} "
                         f"lat={row['latency_s']:5.1f}s tok~{row['tokens_est']}")
        lines.append("")
        lines.append(f"[dim]total {result.total_latency_s:.1f}s · "
                     f"~{result.total_tokens_est} tokens[/dim]")
        if result.memory_hits:
            lines.extend(["", "[b]Memory hits[/b]", ""])
            for h in result.memory_hits:
                lines.append(f"  [dim]({h['score']:.2f})[/dim] {h['question'][:80]}")
        right_body.update("\n".join(lines))

        status.update(
            f"Done. {format_verified_summary(result)} · "
            f"{result.total_latency_s:.1f}s · ~{result.total_tokens_est} tokens."
        )


def main() -> int:
    app = EngineTUI()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
