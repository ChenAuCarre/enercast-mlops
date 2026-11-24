#!/usr/bin/env bash
# Smoke-test a running API: wait for readiness, check the loaded model version, run a real prediction.
# Usage: scripts/smoke_test.sh BASE_URL [EXPECTED_VERSION]
# Env: SMOKE_TIMEOUT (seconds to wait for /health, default 120).
set -euo pipefail

base_url=${1:?usage: smoke_test.sh BASE_URL [EXPECTED_VERSION]}
expected=${2:-}
deadline=$((SECONDS + ${SMOKE_TIMEOUT:-120}))

until curl -sf "$base_url/health" >/dev/null; do
  if ((SECONDS >= deadline)); then
    echo "::error::$base_url/health not ready after ${SMOKE_TIMEOUT:-120}s" >&2
    exit 1
  fi
  sleep 4
done

version=$(curl -sf "$base_url/model-info" | jq -r .resolved_version)
echo "Loaded model version: $version"
if [[ -n $expected && $version != "$expected" ]]; then
  echo "::error::expected model version $expected, got $version" >&2
  exit 1
fi

curl -sf -X POST "$base_url/predict" -H 'Content-Type: application/json' -d '{
  "date": "2016-05-15 19:30:00", "lights": 20,
  "T1": 22.5, "RH_1": 41.2, "T2": 21.8, "RH_2": 40.5,
  "T3": 23.1, "RH_3": 39.8, "T4": 21.4, "RH_4": 38.9,
  "T5": 20.2, "RH_5": 48.1, "T6": 14.3, "RH_6": 62.0,
  "T7": 20.8, "RH_7": 37.5, "T8": 22.0, "RH_8": 42.1,
  "T9": 20.5, "RH_9": 41.0, "To": 13.8, "Pressure": 756.4,
  "RH_out": 65.0, "Windspeed": 4.2, "Visibility": 40.0, "Tdewpoint": 7.3
}' | jq -e '.predicted_appliances_wh | numbers'
