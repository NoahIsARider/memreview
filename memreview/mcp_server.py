#!/usr/bin/env python3
"""memreview MCP server — let any MCP client (Claude Desktop, Claude Code,
Trae, Cursor, ...) read, write, and review your local memory.

Local-first by design: this server runs on *your* machine over stdio and
touches only files under MEMREVIEW_HOME. No network, no cloud, no vendor.

Run:
    pip install -e ".[mcp-server]"
    memreview-mcp            # stdio server — point your MCP client at this

Or without installing:
    python -m memreview.mcp_server
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

from . import config

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # mcp SDK not installed — give a clear hint
    sys.stderr.write(
        "memreview-mcp requires the MCP SDK. Install it with:\n"
        "    pip install -e \".[mcp-server]\"   # or: pip install mcp\n")
    sys.exit(1)

mcp = FastMCP("memreview")


# ─── Path safety: never let a tool read/write outside the memory roots ──────

def _allowed_roots():
    roots = [os.path.realpath(config.NOTES_DIR)]
    for src in config.get_sources():
        roots.append(os.path.realpath(src))
    return roots


def _resolve(rel_path: str) -> str:
    """Resolve a user-supplied path against the notes dir; reject escapes.
    Forgiving: a leading 'notes/' prefix (relative to HOME) is stripped, so
    both '2026-08-21.md' and 'notes/2026-08-21.md' work."""
    p = rel_path
    notes_name = os.path.basename(config.NOTES_DIR.rstrip(os.sep))
    if p.startswith(notes_name + "/"):
        p = p[len(notes_name) + 1:]
    target = os.path.realpath(os.path.join(config.NOTES_DIR, p))
    for root in _allowed_roots():
        if target == root or target.startswith(root + os.sep):
            return target
    raise ValueError(
        f"path escapes the memory roots (notes dir + configured sources): "
        f"{rel_path!r}")


# ─── Tools ───────────────────────────────────────────────────────────────────

@mcp.tool()
def status() -> str:
    """Overview of the memreview setup: home, indexed sources, SRS stats,
    latest context snapshot. Call this first to see what the server knows."""
    config.ensure_dirs()
    lines = [f"🧠 memreview — {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    lines.append(f"   HOME: {config.HOME}")
    lines.append(f"   Sources: {', '.join(config.get_sources()) or '(none)'}")
    try:
        from . import indexer
        idx = indexer.load_index()
        lines.append(f"   Indexed chunks: {idx['meta'].get('total', len(idx['items']))}")
    except Exception:
        pass
    try:
        s = _srs_stats()
        lines.append(f"   SRS: {s['total']} items · {s['due_today']} due today · "
                     f"{s['graduated']} graduated")
    except Exception:
        pass
    try:
        from . import context
        ctx = context._latest()
        if ctx:
            lines.append(f"   Last snapshot: {ctx.get('type')} @ {ctx.get('saved_at', '?')[:19]}")
    except Exception:
        pass
    return "\n".join(lines)


@mcp.tool()
def memory_search(query: str, n: int = 5, source: str = "") -> str:
    """Semantic search over your indexed markdown memory (local embeddings).
    Args: query (required), n (max results, default 5), source (optional
    source tag filter, e.g. 'notes' or 'journal')."""
    from . import indexer
    results = indexer.search(query, n=n, source_filter=source or None)
    if not results:
        return "(no results — run index_rebuild first if the index is empty)"
    return indexer.format_results(results)


@mcp.tool()
def memory_write(filename: str, content: str, append: bool = True) -> str:
    """Write or append a memory note under the notes dir (e.g. filename=
    'ideas/2026-08-21.md'). Append by default; set append=False to overwrite.
    The note is plain Markdown and will be picked up by the next index."""
    target = _resolve(filename)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    mode = "a" if append and os.path.exists(target) else "w"
    with open(target, mode, encoding="utf-8") as f:
        f.write(content if content.endswith("\n") else content + "\n")
    return f"✅ wrote {os.path.relpath(target, config.HOME)} ({mode})"


@mcp.tool()
def memory_daily(content: str) -> str:
    """Append a line/block to today's daily note (notes/YYYY-MM-DD.md) —
    the standard 'log what happened' pattern. Returns the note path."""
    today = datetime.now().strftime("%Y-%m-%d")
    target = _resolve(f"{today}.md")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    stamp = datetime.now().strftime("%H:%M")
    with open(target, "a", encoding="utf-8") as f:
        f.write(f"\n- [{stamp}] {content.rstrip()}\n")
    return f"✅ appended to {os.path.relpath(target, config.HOME)}"


@mcp.tool()
def memory_read(path: str) -> str:
    """Read a memory file (path relative to the notes dir, or absolute within
    the configured sources). Returns the raw Markdown, truncated to 8000
    chars."""
    target = _resolve(path)
    if not os.path.exists(target):
        return f"❌ not found: {path}"
    with open(target, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if len(text) > 8000:
        text = text[:8000] + "\n…[truncated]"
    return text


@mcp.tool()
def context_save(task: str = "", kind: str = "task-switch") -> str:
    """Save a context snapshot (task switch or pre-reset). Use this before
    switching tasks so a later session can restore what you were doing."""
    from . import context
    if kind == "pre-reset":
        fpath = context.save_reset()
    else:
        fpath = context.save(task)
    return f"✅ context saved: {fpath}"


@mcp.tool()
def context_restore(full: bool = False) -> str:
    """Restore the latest context snapshot — a session-start summary by
    default; pass full=True for the complete dump."""
    from . import context
    return context.full_dump() if full else context.summary()


@mcp.tool()
def contexts_list() -> str:
    """List saved context snapshots (newest last)."""
    from . import context
    return context.list_contexts()


@mcp.tool()
def srs_add(category: str, front: str, back: str, example: str = "") -> str:
    """Add a spaced-repetition item (like a flashcard). Category examples:
    'english', 'AI', 'software-engineering', 'management', 'neuroscience'.
    First review is scheduled for tomorrow; intervals 1→3→7→14→30 days."""
    from . import srs
    item = srs.add(category, front, back, example)
    return (f"✅ added {item['id']} ({item['category']}): {item['front']} "
            f"— first review {item['next_review']}")


@mcp.tool()
def srs_due() -> str:
    """List SRS items due for review today (the forgetting-curve push).
    Review them with srs_review."""
    from . import srs
    items = srs.due()
    if not items:
        return "🎉 No items due today."
    return srs.format_due(items)


@mcp.tool()
def srs_review(item_id: str, correct: bool = True) -> str:
    """Mark an SRS item as reviewed. correct=True advances it on the
    Ebbinghaus curve; correct=False reschedules it for tomorrow."""
    from . import srs
    item = srs.review(item_id, correct=correct)
    if not item:
        return f"❌ item not found: {item_id}"
    return f"✅ {item['id']} → next review {item['next_review']}"


@mcp.tool()
def srs_stats() -> str:
    """SRS statistics: total items, due today, graduated."""
    s = _srs_stats()
    return (f"total: {s['total']} · due today: {s['due_today']} · "
            f"graduated: {s['graduated']}")


@mcp.tool()
def index_rebuild() -> str:
    """(Re)embed all markdown files under the configured sources. Incremental:
    unchanged files are skipped. Requires the local embedding endpoint
    (Ollama by default, MEMREVIEW_EMBED_URL to override)."""
    from . import indexer
    new, total = indexer.index_sources()
    return f"✅ index: +{new} new chunks, {total} total"


def _srs_stats():
    from . import srs
    return srs.stats()


def main():
    mcp.run()  # stdio transport — local-first, no network


if __name__ == "__main__":
    main()
