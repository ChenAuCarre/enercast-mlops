#!/usr/bin/env python3
"""Replay CSV rows against a running WattWise API's /predict, one X-Request-ID each.

Usage:
    python scripts/replay.py --file data/versions/v2/train.csv --rate 5 --limit 100
"""

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter

# Raw UCI headers, renamed as in ml/preprocess.py; the API ignores extra columns.
RAW_COLUMN_NAMES = {"T_out": "To", "Press_mm_hg": "Pressure"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="CSV of observations to replay")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--rate", type=float, default=20, help="requests per second")
    parser.add_argument("--limit", type=int, default=1000, help="rows to send")
    args = parser.parse_args()

    with open(args.file, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error(f"no rows in {args.file}")
    random.shuffle(rows)  # a few days of time-ordered rows read as drift

    statuses = Counter()
    for row in rows[: args.limit]:
        payload = {
            RAW_COLUMN_NAMES.get(key, key): value if key == "date" else float(value)
            for key, value in row.items()
            if value and key != "Appliances"  # a missing value -> 422, counted
        }
        request = urllib.request.Request(
            args.url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": uuid.uuid4().hex[:12],
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                statuses[response.status] += 1
        except urllib.error.HTTPError as error:
            statuses[error.code] += 1
        except urllib.error.URLError:
            statuses["unreachable"] += 1
        time.sleep(1 / args.rate)

    print(json.dumps({"responses_by_status": statuses}))


if __name__ == "__main__":
    main()
