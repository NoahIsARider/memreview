# memreview 🧠

**Local-first Markdown memory with spaced-repetition review for AI agents.**

Your agent's memory should do more than store and retrieve — it should
*remember*, which means it must also **review**. memreview combines three ideas:

1. **Markdown is the source of truth.** Memories are plain `.md` files you can
   read, edit, and version — no database, no lock-in, no vendor.
2. **Local semantic search.** Files are embedded with a local embedding model
   (Ollama by default) into an incremental, filterable index. Private by
   design — embeddings never leave your machine.
3. **Ebbinghaus spaced repetition.** Memories that are never revisited decay.
   memreview schedules every item on a 1 → 3 → 7 → 14 → 30 day curve and
   actively pushes due items back into your (or your agent's) attention —
   so long-term memory is *maintained*, not just stored.

Built from a memory system that has run continuously for a year in a real
agent workspace (daily notes, long-term memory file, skill library, and a
knowledge base all indexed and reviewed together).

## Why another memory tool?

The "store + retrieve" half of agent memory is crowded (mem0, EverOS, Basic
Memory...). The **review** half is empty: no mainstream tool schedules memory
revision on a forgetting curve. Retrieval only finds what you search for;
repetition is what actually moves information into long-term memory. memreview
is built around that gap.

## Quick start

```bash
pip install -e .            # or: pip install numpy && clone this repo

export MEMREVIEW_HOME=~/.memreview   # optional; defaults here
memreview index             # embed all .md files under $MEMREVIEW_HOME/notes
memreview search "what was the OOM experiment setup?"
memreview add "ML" "WAL vs DELETE" "SQLite WAL: 10-100x faster concurrent writes" "wal_mode=wal"
memreview review            # show items due today (Ebbinghaus schedule)
memreview review <id> y     # mark correct → next interval
memreview review <id> n     # mark wrong → reschedule tomorrow
```

### Point it at your own notes

```bash
export MEMREVIEW_SOURCES="$HOME/notes:$HOME/journal:$HOME/skills"
memreview index
memreview search "meeting notes about the budget" --source journal
```

### Session continuity for agents

```bash
memreview save --task "switching from experiment A to paper writing"
memreview save --reset                      # full pre-reset dump
memreview restore                           # session-start summary
memreview restore --full
memreview restore --search "what was I doing with the GPU server"
```

## How it works

```
        Markdown files                index.json                queries
  ┌─────────────────────┐     ┌──────────────────────┐    ┌─────────────┐
  │ notes/*.md          │ ──► │ chunk → embed (local)│◄── │ search "..."│
  │ journal/*.md        │     │ cosine similarity    │    └─────────────┘
  │ skills/*.md         │     └──────────────────────┘
  └─────────────────────┘
        + SRS store (items.json) ──► due-today review on 1/3/7/14/30 days
```

- **Incremental indexing** — only new/changed files are embedded (id = hash of
  source+name+chunk).
- **Chunking** — long files are split into 800-char overlapping windows so a
  search can find a needle in a haystack.
- **Filterable** — each source directory becomes a tag (`--source`).
- **Zero servers** — a Python script + Ollama (or any `/api/embeddings`
  endpoint you point `MEMREVIEW_EMBED_URL` at).

## Configuration (all via env vars)

| variable | default | meaning |
|---|---|---|
| `MEMREVIEW_HOME` | `~/.memreview` | index, contexts, SRS store |
| `MEMREVIEW_SOURCES` | `$HOME/notes` | colon-separated dirs of `.md` files |
| `MEMREVIEW_EMBED_URL` | `http://localhost:11434/api/embeddings` | embedding endpoint |
| `MEMREVIEW_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `MEMREVIEW_TZ_OFFSET_HOURS` | `8` | SRS day boundary |

## Project layout

```
memreview/
├── memreview/
│   ├── indexer.py    # semantic index + search (Ollama embeddings)
│   ├── context.py    # context snapshots: save / restore / reset dumps
│   ├── srs.py        # spaced-repetition engine (1/3/7/14/30 days)
│   ├── cli.py        # `memreview` command
│   └── config.py     # env-driven paths & embedding config
├── pyproject.toml
└── LICENSE
```

## Roadmap

- [x] Semantic index + incremental embedding
- [x] Context snapshots (task-switch & pre-reset)
- [x] SRS review engine (Ebbinghaus schedule)
- [ ] OpenAI-compatible embedding client (beyond raw JSON endpoint)
- [ ] `memreview import` from Obsidian / logseq folders
- [ ] Review CLI with inline grading prompts
- [ ] MCP server (`memreview-mcp`) for agents that speak MCP

## License

MIT
