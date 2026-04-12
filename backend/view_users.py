from __future__ import annotations

import argparse
import json

from pymongo import DESCENDING

from db import format_db_target, get_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Print user documents from MongoDB")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of users to print")
    args = parser.parse_args()

    print(f"MongoDB target: {format_db_target()}")
    rows = list(
        get_collection("users").find(
            {},
            {"_id": 0, "password": 0, "otp": 0},
        ).sort([("created_at", DESCENDING), ("email", 1)]).limit(max(1, int(args.limit)))
    )
    for row in rows:
        print(json.dumps(row, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
