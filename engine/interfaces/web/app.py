"""engine.interfaces.web.app — local browser GUI.

FastAPI + HTMX + Jinja2. Runs on http://127.0.0.1:8080 by default. No auth,
no cloud, no analytics. Answer streams into the page via HTMX's out-of-band
swap when the pipeline finishes.

Usage:
    python -m uvicorn engine.interfaces.web.app:app --host 127.0.0.1 --port 8080

Endpoints:
    GET  /                       — the research workbench
    POST /ask                    — run a query, return rendered answer + sources + trace
    POST /memory/reset           — wipe the persistent memory store
    GET  /memory/count           — count of stored trajectories
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from fastapi import FastAPI, Form, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError:  # pragma: no cover
    print(
        "[engine.web] FastAPI is not installed. Run `pip install -r engine/requirements.txt`.",
        file=sys.stderr,
    )
    raise SystemExit(1)

from engine import __version__ as _ENGINE_VERSION  # noqa: E402
from engine.core.memory import MemoryStore  # noqa: E402
from engine.interfaces.common import (  # noqa: E402
    format_sources,
    format_trace_per_node,
    format_verified_summary,
    run_query,
)


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="engine", version=_ENGINE_VERSION)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {
        "request": request,
        "version": _ENGINE_VERSION,
    })


@app.post("/ask", response_class=HTMLResponse)
async def ask(
    request: Request,
    question: str = Form(""),
    domain: str = Form("general"),
    memory: str = Form("session"),
) -> HTMLResponse:
    q = question.strip()
    if not q:
        return HTMLResponse("<div class='err'>Empty question.</div>", status_code=400)

    store = MemoryStore.open(memory) if memory in ("off", "session", "persistent") else None
    result = run_query(q, domain=domain, memory=store)
    if store is not None:
        store.close()

    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": result,
        "verified_summary": format_verified_summary(result),
        "sources": format_sources(result),
        "trace_rows": format_trace_per_node(result),
    })


@app.post("/memory/reset")
async def memory_reset() -> JSONResponse:
    store = MemoryStore.open("persistent")
    n = store.count()
    store.reset()
    store.close()
    return JSONResponse({"reset": n})


@app.get("/memory/count")
async def memory_count() -> JSONResponse:
    store = MemoryStore.open("persistent")
    n = store.count()
    store.close()
    return JSONResponse({"count": n})
