#!/usr/bin/env python3
"""memreview CLI.

Usage:
  memreview index                 — (re)index all markdown sources
  memreview search <query>        — semantic search
  memreview save --task "..."     — snapshot current context (task switch)
  memreview save --reset          — full pre-reset context dump
  memreview restore               — session-start summary
  memreview restore --full        — full context dump
  memreview restore --search Q    — search indexed notes
  memreview add <cat> <front> <back> [example] — add SRS item
  memreview review                — show due SRS items
  memreview review <id> [y|n]     — mark item reviewed (y=correct / n=reschedule)
  memreview stats                 — SRS stats
  memreview contexts              — list saved context snapshots
"""
import argparse
import sys

from . import __version__, context, indexer, srs


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="memreview", description=__doc__)
    p.add_argument("--version", action="version", version=f"memreview {__version__}")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("index", help="(re)index all markdown sources")
    sp.add_argument("--source", default=None, help="filter by source tag")

    sp = sub.add_parser("search", help="semantic search")
    sp.add_argument("query", nargs="+")
    sp.add_argument("-n", type=int, default=8)
    sp.add_argument("--source", default=None)

    sp = sub.add_parser("save", help="save a context snapshot")
    sp.add_argument("--task", default="")
    sp.add_argument("--reset", action="store_true")

    sp = sub.add_parser("restore", help="restore context")
    sp.add_argument("--full", action="store_true")
    sp.add_argument("--search", nargs="+", default=None)

    sp = sub.add_parser("add", help="add an SRS item")
    sp.add_argument("category")
    sp.add_argument("front")
    sp.add_argument("back")
    sp.add_argument("example", nargs="?", default="")

    sp = sub.add_parser("review", help="show/mark due SRS items")
    sp.add_argument("item_id", nargs="?", default=None)
    sp.add_argument("correct", nargs="?", choices=["y", "n"], default="y")

    sp = sub.add_parser("stats", help="SRS stats")
    sp = sub.add_parser("contexts", help="list saved snapshots")

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 0

    if args.cmd == "index":
        new, total = indexer.index_sources()
        print(f"✅ done: +{new} new, {total} total")
    elif args.cmd == "search":
        query = " ".join(args.query)
        results = indexer.search(query, n=args.n, source_filter=args.source)
        print(indexer.format_results(results) if results else "(no results)")
    elif args.cmd == "save":
        fpath = context.save_reset() if args.reset else context.save(args.task)
        print(f"✅ context saved: {fpath}")
    elif args.cmd == "restore":
        if args.search:
            print(context.search_memories(" ".join(args.search)))
        elif args.full:
            print(context.full_dump())
        else:
            print(context.summary())
    elif args.cmd == "add":
        item = srs.add(args.category, args.front, args.back, args.example)
        print(f"✅ added {item['id']} ({item['category']}), first review "
              f"{item['next_review']}")
    elif args.cmd == "review":
        if args.item_id:
            item = srs.review(args.item_id, correct=(args.correct == "y"))
            if item:
                print(f"✅ {item['id']} → next review {item['next_review']}")
            else:
                print(f"❌ item not found: {args.item_id}")
        else:
            print(srs.format_due(srs.due()))
    elif args.cmd == "stats":
        s = srs.stats()
        print(f"total: {s['total']} · due today: {s['due_today']} · graduated: {s['graduated']}")
    elif args.cmd == "contexts":
        print(context.list_contexts())
    return 0


if __name__ == "__main__":
    sys.exit(main())
