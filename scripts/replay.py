#!/usr/bin/env python3
"""Replay prediction traffic against a running WattWise API.

Sends rows of a CSV (the ``date`` column plus the model feature columns) to the
``/predict`` endpoint, with an ``X-Request-ID`` per request, and reports the
response statuses.

Usage:
    python scripts/replay.py --file data/versions/v2/train.csv --url http://localhost:8000/predict
    python scripts/replay.py --file data/versions/v2/train.csv --rate 5 --limit 100
    python scripts/replay.py --file data/versions/v2/train.csv --broken
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter

FEATURE_COLUMNS = (
    "lights",
    *(f"T{i}" for i in range(1, 10)),
    *(f"RH_{i}" for i in range(1, 10)),
    "To",
    "Pressure",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="CSV of observations to replay")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--rate", type=float, default=20, help="requests per second")
    parser.add_argument("--limit", type=int, default=1000, help="number of rows to send")
    parser.add_argument("--shuffle", action="store_true", help="send rows in random order")
    parser.add_argument("--broken", action="store_true", help="send rows with a missing column")
    args = parser.parse_args()

    with open(args.file, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if args.shuffle:
        rows = rows[::-1]

    fields = ["date", *FEATURE_COLUMNS]
    rows = [row for row in rows if all(field in row for field in fields)][: args.limit]

    statuses: Counter = Counter()
    for row in rows:
        payload = {field: float(row[field]) for field in FEATURE_COLUMNS}
        payload["date"] = row["date"]

        if args.broken:
            payload.pop(FEATURE_COLUMNS[0], None)

        request = urllib.request.Request(
            args.url,
            data=json.dumps(payload).encode(),
            method="POST",
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

    print(json.dumps({"responses_by_status": dict(statuses)}))


if __name__ == "__main__":
    main()
