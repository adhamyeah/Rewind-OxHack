"""
Runs the demo queries in queries.sql against Tiger Data and prints readable tables.

  python show_history.py                  # all queries
  python show_history.py --label keys     # the timeline query for another item
  python show_history.py --only 4         # just query number 4
"""

import argparse
import datetime as dt
import re

import config


def load_queries(path="queries.sql"):
    """[(title, sql)] from the '-- name:' sections of queries.sql."""
    text = open(path).read()
    parts = re.split(r"^-- name:\s*(.+)$", text, flags=re.MULTILINE)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def show(value):
    if isinstance(value, dt.datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return "-" if value is None else str(value)


def print_table(columns, rows):
    if not rows:
        print("  (no rows)")
        return
    cells = [[show(v) for v in row] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    line = "  " + "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    print(line)
    print("  " + "  ".join("-" * w for w in widths))
    for r in cells:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(r, widths)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="glasses", help="item for the timeline query")
    ap.add_argument("--only", type=int, default=0, help="run only this query number (1-based)")
    args = ap.parse_args()

    if not config.TIGER_DATABASE_URL:
        print("TIGER_DATABASE_URL is not set in .env")
        return
    import psycopg

    try:
        conn = psycopg.connect(config.TIGER_DATABASE_URL, autocommit=True, connect_timeout=8)
    except Exception as e:
        print(f"Could not connect to Tiger Data: {e}")
        return
    with conn, conn.cursor() as cur:
        try:  # bring the continuous aggregate up to date first
            cur.execute("CALL refresh_continuous_aggregate('rewind_events_5min', NULL, NULL)")
        except Exception as e:
            print(f"(could not refresh the continuous aggregate: {e})")
        for number, (title, sql) in enumerate(load_queries(), 1):
            if args.only and number != args.only:
                continue
            print(f"\n{number}. {title}")
            try:
                cur.execute(sql, {"label": args.label})
                print_table([d.name for d in cur.description], cur.fetchall())
            except Exception as e:
                print(f"  query failed: {e}")


if __name__ == "__main__":
    main()
