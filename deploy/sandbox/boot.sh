#!/usr/bin/env bash
# Boot the full Durable Training stack inside a Daytona sandbox.
#
# The launcher execs this with TEMPORAL_UI_PROXY_URL set to the *browser-reachable*
# signed preview URL of port 8234, so the dashboard can hand the iframe a URL that
# resolves from outside the sandbox.
set -euo pipefail
cd /app

echo "[boot] starting Temporal dev server..."
temporal server start-dev --port 7233 --ui-port 8233 \
  >/tmp/temporal.log 2>&1 &

echo "[boot] waiting for Temporal UI on :8233..."
for _ in $(seq 1 90); do
  curl -sf http://localhost:8233/ >/dev/null 2>&1 && break
  sleep 1
done

# The dashboard API's lifespan also starts the UI proxy (on PROXY_HOST:8234) and
# one worker. Bind everything browser-facing on 0.0.0.0 so Daytona can preview it.
export PROXY_HOST=0.0.0.0
export TEMPORAL_UI_PROXY_URL="${TEMPORAL_UI_PROXY_URL:-http://localhost:8234}"

echo "[boot] starting dashboard API on :8000 (UI proxy -> ${TEMPORAL_UI_PROXY_URL})"
exec uv run uvicorn dashboard.api.main:app --host 0.0.0.0 --port 8000
