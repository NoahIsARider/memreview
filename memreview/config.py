"""Configuration — everything is overridable via environment variables."""
import os

# Root data directory (index, contexts, SRS store)
HOME = os.environ.get("MEMREVIEW_HOME", os.path.expanduser("~/.memreview"))
INDEX_FILE = os.environ.get("MEMREVIEW_INDEX", os.path.join(HOME, "index.json"))
CONTEXT_DIR = os.path.join(HOME, "contexts")
SRS_FILE = os.environ.get("MEMREVIEW_SRS", os.path.join(HOME, "srs", "items.json"))
NOTES_DIR = os.path.join(HOME, "notes")

# Embedding backend (Ollama by default; point MEMREVIEW_EMBED_URL at any
# OpenAI-compatible /api/embeddings endpoint that accepts {"model","prompt"})
EMBED_URL = os.environ.get("MEMREVIEW_EMBED_URL", "http://localhost:11434/api/embeddings")
EMBED_MODEL = os.environ.get("MEMREVIEW_EMBED_MODEL", "nomic-embed-text")

# Index sources: colon-separated list of directories whose .md files are indexed.
# Each top-level directory name becomes the item's "source" tag.
# Default: the notes directory under MEMREVIEW_HOME.
def get_sources():
    raw = os.environ.get("MEMREVIEW_SOURCES", "")
    if raw.strip():
        return [d for d in raw.split(":") if d.strip()]
    return [NOTES_DIR]


def ensure_dirs():
    for d in (HOME, CONTEXT_DIR, os.path.dirname(SRS_FILE), NOTES_DIR):
        os.makedirs(d, exist_ok=True)
