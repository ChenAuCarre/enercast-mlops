import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replay.py"
RAW_HEADER = [
    "date",
    "Appliances",
    "lights",
    *(col for i in range(1, 10) for col in (f"T{i}", f"RH_{i}")),
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
]


def _replay(tmp_path, rows, *args):
    path = tmp_path / "rows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RAW_HEADER)
        writer.writerows(rows)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--file", str(path), "--rate", "1000"]
        + ["--url", "http://127.0.0.1:9/predict", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_replay_sends_rows_with_raw_uci_headers(tmp_path):
    row = ["2016-01-11 17:00:00", *["1.0"] * (len(RAW_HEADER) - 1)]

    result = _replay(tmp_path, [row, row])

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"responses_by_status": {"unreachable": 2}}


def test_replay_fails_when_no_row_is_complete(tmp_path):
    row = ["2016-01-11 17:00:00", *["1.0"] * (len(RAW_HEADER) - 2), ""]

    result = _replay(tmp_path, [row])

    assert result.returncode != 0
    assert "no complete rows" in result.stderr


def test_replay_rejects_a_shift_without_a_number(tmp_path):
    result = _replay(tmp_path, [], "--shift", "To")

    assert result.returncode != 0
    assert "not FEATURE=DELTA" in result.stderr
