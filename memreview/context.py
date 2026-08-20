"""Context snapshots — save and restore agent context across sessions.

Two workflows:
  - `save --task "..."` : quick snapshot when switching tasks (state + previews)
  - `save --reset`      : full pre-reset dump (full notes + MEMORY + instructions)
  - `restore`           : print a session-start summary of the latest snapshot
  - `restore --full`    : dump everything
  - `restore --search Q`: semantic search over indexed notes
"""
import glob
import json
import os
import subprocess
from datetime import datetime

from . import config


def _ensure():
    config.ensure_dirs()


# ─── Save ──────────────────────────────────────────────────

def _collect_state():
    state = {}
    # Active files: previews of key files in the workspace
    state["active_files"] = []
    for pattern in config.get_sources() + [os.path.join(config.HOME, "notes")]:
        for fpath in sorted(glob.glob(f"{pattern}/*.md")):
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    state["active_files"].append({
                        "path": os.path.relpath(fpath, config.HOME),
                        "preview": f.read(200)})
            except Exception:
                pass
    return state


def save(task_description="", kind="task-switch", notes=None):
    """Save a context snapshot; returns the file path."""
    _ensure()
    state = _collect_state()
    ctx = {
        "type": kind,
        "saved_at": datetime.now().isoformat(),
        "task_description": task_description,
        "notes": notes or "",
        "workspace_state": state,
    }
    ts = ctx["saved_at"].replace(":", "-").replace("T", "_")[:19]
    fpath = os.path.join(config.CONTEXT_DIR, f"{ts}_{kind}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False)
    return fpath


def save_reset(memory_file=None):
    """Full pre-reset dump: includes whole notes dir contents inline."""
    _ensure()
    notes = {}
    for src_dir in config.get_sources():
        if not os.path.isdir(src_dir):
            continue
        source = os.path.basename(os.path.normpath(src_dir))
        for fpath in sorted(glob.glob(f"{src_dir}/**/*.md", recursive=True)):
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                notes[f"{source}/{os.path.relpath(fpath, src_dir)}"] = f.read()
    ctx = {
        "type": "pre-reset",
        "saved_at": datetime.now().isoformat(),
        "workspace_state": _collect_state(),
        "full_notes": notes,
        "reset_instructions": [
            "1. Read this context file first",
            "2. Read the most recent daily note",
            "3. `memreview search '<topic>'` for additional context",
            "4. Resume from the 'next_action' field",
        ],
        "next_action": "fill this in with what to do next",
    }
    ts = ctx["saved_at"].replace(":", "-").replace("T", "_")[:19]
    fpath = os.path.join(config.CONTEXT_DIR, f"{ts}_pre-reset.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False)
    return fpath


def _latest():
    _ensure()
    files = sorted(glob.glob(f"{config.CONTEXT_DIR}/*.json"), reverse=True)
    if not files:
        return None
    with open(files[0], encoding="utf-8") as f:
        return json.load(f)


# ─── Restore ───────────────────────────────────────────────

def summary():
    """Session-start summary from the latest snapshot."""
    ctx = _latest()
    lines = [f"memreview — {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    if not ctx:
        lines.append("   No saved contexts yet.")
        lines.append("   Try: memreview save --task 'what I'm working on'")
        return "\n".join(lines)
    lines.append(f"   Latest snapshot: {ctx.get('type')} @ {ctx.get('saved_at', '?')[:19]}")
    task = ctx.get("task_description") or ctx.get("next_action") or "(no task)"
    lines.append(f"   Task: {task[:120]}")
    ws = ctx.get("workspace_state", {})
    files = ws.get("active_files", [])
    if files:
        lines.append(f"   Active files ({len(files)}): "
                     + ", ".join(f["path"] for f in files[:8]))
    if ctx.get("type") == "pre-reset":
        lines.append("   ⚠️  pre-reset dump — full context available (`restore --full`)")
    lines.append("   💡 memreview search '<topic>' · memreview restore --full")
    return "\n".join(lines)


def full_dump():
    ctx = _latest()
    out = ["=" * 60, "  MEMREVIEW — FULL CONTEXT DUMP", f"  {datetime.now().isoformat()}", "=" * 60]
    if ctx:
        out.append("📋 LATEST SNAPSHOT")
        out.append(json.dumps(ctx, indent=2, ensure_ascii=False)[:4000])
    return "\n".join(out)


def search_memories(query, n=5):
    try:
        from . import indexer
        results = indexer.search(query, n=n)
        return indexer.format_results(results) if results else "(no results)"
    except Exception as e:
        return f"Search failed: {e}"


def list_contexts():
    _ensure()
    files = sorted(glob.glob(f"{config.CONTEXT_DIR}/*.json"))
    if not files:
        return "No saved contexts."
    lines = []
    for i, f in enumerate(files[-20:], 1):
        try:
            with open(f, encoding="utf-8") as fh:
                ctx = json.load(fh)
            lines.append(f"{i:>3}  {ctx.get('saved_at', '?')[:19]}  "
                         f"{ctx.get('type', '?'):<12}  "
                         f"{str(ctx.get('task_description', ''))[:40]}")
        except Exception:
            pass
    return "\n".join(lines)


# Keep subprocess-based search for backwards compatibility
def _search_subprocess(query, n=5):
    try:
        result = subprocess.run(
            ["python3", "-m", "memreview.indexer", query, "--n", str(n)],
            capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else f"failed: {result.stderr[:200]}"
    except Exception as e:
        return f"error: {e}"
