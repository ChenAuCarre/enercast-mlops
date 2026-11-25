"""Replay CSV rows against /predict to exercise the monitoring dashboard and alerts.

Usage: uv run python scripts/replay.py data/versions/test/final_test.csv \
    [--url http://localhost:8000] [--delay 0.05]
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.request

RENAME = {"T_out": "To", "Press_mm_hg": "Pressure"}  # dataset -> API names


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.csv, newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream), 1):
            body = {
                RENAME.get(k, k): v if k == "date" else float(v) for k, v in row.items()
            }
            request = urllib.request.Request(
                f"{args.url}/predict",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request) as response:
                    status = response.status
            except urllib.error.HTTPError as exc:
                status = exc.code
            print(i, status, flush=True)
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
