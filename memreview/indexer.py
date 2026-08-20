"""Semantic indexer — embed Markdown files and search them locally.

Indexes every .md file under the configured source directories. Each source
directory's basename becomes the item's `source` tag, so you can filter by
kind (e.g. notes/, skills/, journal/). Embeddings are computed locally via
Ollama by default; incremental (only new/changed files are embedded).
"""
import glob
import hashlib
import json
import os
import time
from datetime import datetime

import numpy as np
import urllib.request

from . import config


# ─── Embedding ─────────────────────────────────────────────

def embed(text, retries=2):
    """Get an embedding from the configured endpoint (Ollama by default)."""
    data = json.dumps({"model": config.EMBED_MODEL, "prompt": text[:2000]}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                config.EMBED_URL, data=data,
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())["embedding"]
        except Exception:
            if attempt < retries:
                time.sleep(1)
                continue
            raise


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ─── Index persistence ─────────────────────────────────────

def load_index():
    if os.path.exists(config.INDEX_FILE):
        with open(config.INDEX_FILE) as f:
            return json.load(f)
    return {"items": [], "meta": {"last_indexed": None, "total": 0}}


def save_index(idx):
    os.makedirs(os.path.dirname(config.INDEX_FILE), exist_ok=True)
    idx["meta"]["total"] = len(idx["items"])
    idx["meta"]["last_indexed"] = datetime.now().isoformat()
    with open(config.INDEX_FILE, "w") as f:
        json.dump(idx, f, indent=2)


def make_item_id(source, name):
    return hashlib.md5(f"{source}:{name}".encode()).hexdigest()[:16]


def chunk_text(text, max_chars=800, overlap=100):
    """Split long text into overlapping chunks for better embedding."""
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks


# ─── Indexing ──────────────────────────────────────────────

def _index_file(idx, source, name, content, existing):
    """Embed one file (possibly chunked); returns count of new items."""
    new_count = 0
    chunks = chunk_text(content)
    for i, chunk in enumerate(chunks):
        item_id = make_item_id(source, f"{name}.{i}")
        if item_id in existing:
            continue
        try:
            vec = embed(chunk)
        except Exception as e:
            print(f"    ✗ {name}[{i}]: {e}")
            continue
        idx["items"].append({
            "id": item_id, "source": source, "name": name,
            "type": "note", "text": chunk[:500], "embedding": vec,
        })
        existing.add(item_id)
        new_count += 1
    return new_count


def index_sources():
    """Re-index all configured sources. Returns (new_items, total_items)."""
    config.ensure_dirs()
    idx = load_index()
    existing = set(i["id"] for i in idx["items"])
    total_new = 0

    for src_dir in config.get_sources():
        if not os.path.isdir(src_dir):
            print(f"  (skip: {src_dir} does not exist)")
            continue
        source = os.path.basename(os.path.normpath(src_dir))
        files = sorted(glob.glob(f"{src_dir}/**/*.md", recursive=True))
        print(f"📄 {source} — {len(files)} markdown files")
        for fpath in files:
            fname = os.path.relpath(fpath, src_dir)
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
            if not content.strip():
                continue
            n = _index_file(idx, source, fname, content, existing)
            if n:
                print(f"    + {fname} ({n} chunk(s))")
            total_new += n

    save_index(idx)
    return total_new, len(idx["items"])


# ─── Search ────────────────────────────────────────────────

def search(query, n=8, source_filter=None):
    """Semantic search; optional filter by source tag."""
    idx = load_index()
    items = idx["items"]
    if not items:
        return []
    if source_filter:
        items = [i for i in items if i.get("source") == source_filter]
    if not items:
        return []
    q_vec = embed(query)
    scored = sorted(
        ((cosine_sim(q_vec, i["embedding"]), i) for i in items),
        key=lambda x: -x[0])
    return scored[:n]


def format_results(results):
    lines = []
    for i, (score, item) in enumerate(results):
        lines.append(f"[{i+1}] ({score*100:.0f}%) {item.get('source')}/{item.get('name')}")
        lines.append(f"    {item.get('text', '')[:150]}")
    return "\n".join(lines)
