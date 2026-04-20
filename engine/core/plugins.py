"""engine.core.plugins — load external Claude plugins + Hermes skills.

Phase 5 of the master plan. Lets users install third-party Claude
plugins (from GitHub repos, local paths, or marketplace URLs) and
Hermes-format skills (`agentskills.io` YAML frontmatter) into a local
registry. The engine surfaces installed skills to the model via the
plugin registry; MCP servers declared in a plugin's `plugin.json` can
be bridged (Phase 6+) but for now are recorded as metadata.

Safety model:
  - Default-deny: no plugin auto-installs.
  - Source validation: `install(...)` inspects the manifest before
    writing to the registry.
  - Forbidden-symbols scan on any executable hook code (Python / shell)
    reuses engine.core.safety.
  - Registry lives at `~/.agentic-research/plugins/`. Inspectable,
    editable, wipable.

Sources supported in v1:
  - `gh:owner/repo[@ref]`       git clone via shell git; HEAD or specified ref
  - `file:<absolute-path>`       local directory
  - `https://.../marketplace.json` remote manifest (Claude plugin spec)

Registry on disk:
    ~/.agentic-research/plugins/
      index.json               — list of installed plugins + their source
      <plugin-name>/           — plugin contents (.claude-plugin/plugin.json + skills/…)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ENV = os.environ.get
DEFAULT_PLUGINS_DIR = Path(ENV("PLUGINS_DIR", str(Path.home() / ".agentic-research" / "plugins")))


# ── Data structures ─────────────────────────────────────────────────

@dataclass
class PluginManifest:
    """Normalized view of a plugin — covers both Claude plugin.json and
    Hermes skill YAML frontmatter (the latter gets wrapped as a synthetic
    single-skill plugin).
    """

    name: str
    source: str                               # the URI / path installed from
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    skills: list[dict] = field(default_factory=list)   # [{name, path, description, triggers}]
    mcp_servers: list[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict)   # original manifest for reference

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InstalledPlugin:
    """One entry in the registry index."""

    name: str
    version: str
    source: str
    install_path: str
    manifest: dict

    def to_dict(self) -> dict:
        return asdict(self)


# ── Safety ────────────────────────────────────────────────────────────

# Hooks / agents that reference any of these are rejected outright.
FORBIDDEN_SYMBOLS = (
    "eval(",
    "exec(",
    "subprocess.Popen",  # use allow-list instead if you need subprocess
    "__import__",
    "os.system(",
    "shutil.rmtree('/')",
    "/etc/passwd",
)


def scan_for_forbidden(text: str) -> list[str]:
    """Return the list of forbidden substrings that appear in `text` (if any)."""
    lowered = text.lower()
    return [s for s in FORBIDDEN_SYMBOLS if s.lower() in lowered]


# ── Parsers ──────────────────────────────────────────────────────────

def _parse_yaml_frontmatter(text: str) -> tuple[dict, str]:
    """Tiny YAML frontmatter parser. Returns ({}, text) if no frontmatter.

    Only supports the narrow dict-of-scalars + list-of-scalars format that
    agentskills.io uses. No dependency on PyYAML to keep the install light.
    """
    if not text.startswith("---"):
        return {}, text
    try:
        end = text.index("\n---", 3)
    except ValueError:
        return {}, text
    header = text[3:end].strip()
    body = text[end + len("\n---"):].lstrip("\n")

    meta: dict = {}
    current_key: str | None = None
    for line in header.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        if line.startswith(("  - ", "- ", "   - ")):
            # list item under `current_key`
            if current_key is None:
                continue
            meta.setdefault(current_key, []).append(line.strip()[2:].strip())
        elif ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if v == "" or v == "|":
                meta[k] = []
                current_key = k
            else:
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1]
                meta[k] = v
                current_key = k
    return meta, body


def parse_claude_plugin(plugin_dir: Path) -> PluginManifest:
    """Read `<plugin_dir>/.claude-plugin/plugin.json` and normalize."""
    pj = plugin_dir / ".claude-plugin" / "plugin.json"
    if not pj.exists():
        raise FileNotFoundError(f"missing .claude-plugin/plugin.json in {plugin_dir}")
    data: dict[str, Any] = json.loads(pj.read_text())

    skills_list: list[dict] = []
    for skill_rel in data.get("skills", []) or []:
        skill_path = plugin_dir / skill_rel
        if not skill_path.exists():
            continue
        meta, _body = _parse_yaml_frontmatter(skill_path.read_text())
        skills_list.append({
            "name": meta.get("name") or skill_path.stem,
            "path": str(skill_path.relative_to(plugin_dir)),
            "description": meta.get("description", ""),
            "triggers": meta.get("triggers", []),
        })

    mcps_dict = data.get("mcpServers", {}) or {}
    mcps = [{"name": k, "config": v} for k, v in mcps_dict.items()]

    return PluginManifest(
        name=data.get("name") or plugin_dir.name,
        source=str(plugin_dir),
        version=str(data.get("version", "0.0.0")),
        description=str(data.get("description", "")),
        author=(data.get("author") or {}).get("name", "") if isinstance(data.get("author"), dict) else str(data.get("author", "")),
        skills=skills_list,
        mcp_servers=mcps,
        raw=data,
    )


def parse_hermes_skill(skill_path: Path) -> PluginManifest:
    """Wrap a single agentskills.io-format Markdown file as a plugin.

    The Markdown file is copied verbatim into the plugin dir so
    downstream loaders can read it through the same code path.
    """
    meta, _ = _parse_yaml_frontmatter(skill_path.read_text())
    name = meta.get("name") or skill_path.stem
    return PluginManifest(
        name=name,
        source=str(skill_path),
        version=str(meta.get("version", "0.0.0")),
        description=str(meta.get("description", "")),
        author=str(meta.get("author", "")),
        skills=[{
            "name": name,
            "path": skill_path.name,
            "description": meta.get("description", ""),
            "triggers": meta.get("triggers", []),
        }],
        mcp_servers=[],
        raw={"frontmatter": meta, "kind": "hermes-skill"},
    )


# ── PluginRegistry ────────────────────────────────────────────────────

class PluginRegistry:
    """Disk-backed registry of installed plugins / skills.

    Usage:
        reg = PluginRegistry()
        reg.install("gh:owner/repo")
        reg.install("file:/path/to/engine/mcp/claude_plugin")
        reg.list()               # → [InstalledPlugin, …]
        reg.uninstall("name")
        reg.inspect("name")
    """

    def __init__(self, root: Path = DEFAULT_PLUGINS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "index.json"
        if not self.index_file.exists():
            self.index_file.write_text(json.dumps({"plugins": []}, indent=2))

    # ── public API ──────────────────────────────────────────

    def list(self) -> list[InstalledPlugin]:
        data = json.loads(self.index_file.read_text())
        return [InstalledPlugin(**p) for p in data.get("plugins", [])]

    def inspect(self, name: str) -> InstalledPlugin | None:
        for p in self.list():
            if p.name == name:
                return p
        return None

    def install(self, source: str) -> InstalledPlugin:
        """Resolve `source`, fetch into a staging dir, validate, register."""
        kind, payload = _parse_source(source)
        staging = self._stage(kind, payload)
        try:
            # Two manifest shapes supported.
            if (staging / ".claude-plugin" / "plugin.json").exists():
                manifest = parse_claude_plugin(staging)
                self._safety_scan(staging)
            elif staging.is_file() and staging.suffix in {".md", ".markdown"}:
                manifest = parse_hermes_skill(staging)
                self._safety_scan_file(staging)
            else:
                # Maybe it's a directory of hermes skills — take the first .md file.
                md_files = sorted(staging.glob("*.md")) if staging.is_dir() else []
                if not md_files:
                    raise ValueError(
                        f"no plugin.json or .md skill found under {staging}"
                    )
                manifest = parse_hermes_skill(md_files[0])
                self._safety_scan_file(md_files[0])

            target = self.root / manifest.name
            if target.exists():
                shutil.rmtree(target)
            if staging.is_dir():
                shutil.copytree(staging, target)
            else:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy(staging, target / staging.name)

            entry = InstalledPlugin(
                name=manifest.name,
                version=manifest.version,
                source=source,
                install_path=str(target),
                manifest=manifest.to_dict(),
            )
            self._add_to_index(entry)
            return entry
        finally:
            # Clean up staging if we created it
            if staging.exists() and staging.is_dir() and staging.parent.name.startswith(".staging-"):
                shutil.rmtree(staging.parent, ignore_errors=True)

    def uninstall(self, name: str) -> bool:
        """Remove a plugin by name. Returns True if removed."""
        entries = self.list()
        match = next((e for e in entries if e.name == name), None)
        if match is None:
            return False
        path = Path(match.install_path)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        remaining = [e for e in entries if e.name != name]
        self._write_index(remaining)
        return True

    def reset(self) -> int:
        """Wipe all plugins. Returns count removed."""
        entries = self.list()
        for e in entries:
            p = Path(e.install_path)
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        self._write_index([])
        return len(entries)

    # ── internals ───────────────────────────────────────────

    def _stage(self, kind: str, payload: str) -> Path:
        """Copy source content into a staging dir and return its root path."""
        staging_parent = self.root / f".staging-{os.getpid()}-{abs(hash(payload))}"
        if staging_parent.exists():
            shutil.rmtree(staging_parent)
        staging_parent.mkdir(parents=True)

        if kind == "file":
            src = Path(payload).expanduser().resolve()
            if not src.exists():
                raise FileNotFoundError(str(src))
            if src.is_dir():
                dest = staging_parent / src.name
                shutil.copytree(src, dest)
                return dest
            dest = staging_parent / src.name
            shutil.copy(src, dest)
            return dest

        if kind == "gh":
            # payload like "owner/repo@ref" or "owner/repo"
            if "@" in payload:
                slug, ref = payload.split("@", 1)
            else:
                slug, ref = payload, None
            url = f"https://github.com/{slug}.git"
            dest = staging_parent / slug.replace("/", "-")
            cmd = ["git", "clone", "--depth", "1"]
            if ref:
                cmd += ["--branch", ref]
            cmd += [url, str(dest)]
            subprocess.run(cmd, check=True, capture_output=True)
            return dest

        if kind == "url":
            # Fetch a marketplace.json and download referenced plugins.
            # v1 scope: only the marketplace manifest itself is downloaded;
            # per-plugin downloads are delegated to repeated install() calls.
            dest = staging_parent / "marketplace.json"
            with urllib.request.urlopen(payload, timeout=20) as resp:
                dest.write_bytes(resp.read())
            return dest

        raise ValueError(f"unknown source kind: {kind}")

    def _safety_scan(self, plugin_dir: Path) -> None:
        """Walk every text file in the plugin dir; raise if forbidden found."""
        for p in plugin_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".md", ".markdown", ".py", ".sh", ".json", ".yaml", ".yml", ".txt"}:
                continue
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            found = scan_for_forbidden(text)
            if found:
                raise RuntimeError(
                    f"plugin rejected: forbidden symbols {found} found in {p}"
                )

    def _safety_scan_file(self, path: Path) -> None:
        text = path.read_text(errors="replace")
        found = scan_for_forbidden(text)
        if found:
            raise RuntimeError(f"plugin rejected: forbidden symbols {found} found in {path}")

    def _add_to_index(self, entry: InstalledPlugin) -> None:
        entries = [e for e in self.list() if e.name != entry.name]
        entries.append(entry)
        self._write_index(entries)

    def _write_index(self, entries: list[InstalledPlugin]) -> None:
        self.index_file.write_text(json.dumps(
            {"plugins": [e.to_dict() for e in entries]}, indent=2,
        ))


# ── Source parsing ──────────────────────────────────────────────────

def _parse_source(source: str) -> tuple[str, str]:
    """Return (kind, payload). Kinds: gh / file / url."""
    if source.startswith("gh:"):
        return "gh", source[3:]
    if source.startswith("file:"):
        return "file", source[5:]
    if source.startswith(("http://", "https://")):
        return "url", source
    # Default: treat bare paths as file:
    if Path(source).exists():
        return "file", source
    raise ValueError(
        f"unrecognized source {source!r}. "
        f"Use gh:owner/repo, file:/abs/path, or https://url."
    )


__all__ = [
    "DEFAULT_PLUGINS_DIR",
    "FORBIDDEN_SYMBOLS",
    "PluginManifest",
    "InstalledPlugin",
    "PluginRegistry",
    "scan_for_forbidden",
    "parse_claude_plugin",
    "parse_hermes_skill",
]
