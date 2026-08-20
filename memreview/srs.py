"""Spaced-repetition review engine (Ebbinghaus schedule).

Intervals default to 1 → 3 → 7 → 14 → 30 days; after the last interval an
item "graduates". This is what makes memreview a *memory* system rather than
just a search index: memories that are not reviewed decay, so the engine
actively pushes due items back into your (or your agent's) attention.
"""
import json
import os
from datetime import datetime, timedelta, timezone

from . import config

DEFAULT_INTERVALS = [1, 3, 7, 14, 30]
TIMEZONE = timezone(timedelta(hours=8))  # configurable below
TZ_OFFSET = int(os.environ.get("MEMREVIEW_TZ_OFFSET_HOURS", "8"))
TIMEZONE = timezone(timedelta(hours=TZ_OFFSET))


def _load():
    config.ensure_dirs()
    if os.path.exists(config.SRS_FILE):
        with open(config.SRS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"config": {"intervals_days": DEFAULT_INTERVALS}, "items": []}
        _save(data)
    return data


def _save(data):
    os.makedirs(os.path.dirname(config.SRS_FILE), exist_ok=True)
    with open(config.SRS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def today_str():
    return datetime.now(TIMEZONE).strftime("%Y-%m-%d")


def add(category, front, back, example=""):
    """Add an item; first review scheduled for tomorrow."""
    data = _load()
    prefix = category[:4].lower() if category else "item"
    existing = [i for i in data["items"] if i["id"].startswith(prefix)]
    nid = f"{prefix}-{len(existing)+1:03d}"
    item = {
        "id": nid,
        "category": category,
        "front": front,
        "back": back,
        "example": example,
        "reviews": [],
        "next_review": (datetime.now(TIMEZONE) + timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    data["items"].append(item)
    _save(data)
    return item


def due():
    """Items due for review today."""
    data = _load()
    today = today_str()
    return [i for i in data["items"] if i.get("next_review") == today]


def stats():
    data = _load()
    total = len(data["items"])
    due_n = len(due())
    graduated = sum(1 for i in data["items"] if i.get("next_review") == "graduated")
    return {"total": total, "due_today": due_n, "graduated": graduated}


def review(item_id, correct=True):
    """Advance an item after review. `correct=False` reschedules to tomorrow."""
    data = _load()
    intervals = data.get("config", {}).get("intervals_days", DEFAULT_INTERVALS)
    item = next((i for i in data["items"] if i["id"] == item_id), None)
    if item is None:
        return None
    if not correct:
        item["reviews"].append({"date": today_str(), "interval_index": -1})
        item["next_review"] = (datetime.now(TIMEZONE) + timedelta(days=1)).strftime("%Y-%m-%d")
        _save(data)
        return item
    count = len([r for r in item["reviews"] if r.get("interval_index", -1) >= 0])
    item["reviews"].append({"date": today_str(), "interval_index": count})
    if count < len(intervals):
        days = intervals[count]
        item["next_review"] = (datetime.now(TIMEZONE) + timedelta(days=days)).strftime("%Y-%m-%d")
    else:
        item["next_review"] = "graduated"
    _save(data)
    return item


def format_due(items):
    if not items:
        return "🎉 Nothing due today."
    lines = [f"📚 {len(items)} item(s) due today:"]
    for i, item in enumerate(items, 1):
        lines.append(f"\n[{i}] {item['category']} — {item['front']}")
        lines.append(f"    {item['back']}")
        if item.get("example"):
            lines.append(f"    e.g. {item['example']}")
        lines.append(f"    (id: {item['id']})")
    return "\n".join(lines)
