#!/usr/bin/env bash
# Bring up the whole dashboard with one command:
#   - Temporal dev server (reused if already running; otherwise started detached)
#   - the API bridge (also auto-spawns a worker + the UI proxy on :8234)
#   - the SPA dev server
#
# Ctrl-C tears down what this script started; an already-running Temporal server
# is left alone.
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

API_PORT="${API_PORT:-8000}"

PIDS=()

cleanup() {
  echo ""
  echo "Shutting down (Temporal dev server left running)..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  exit 0
}
trap cleanup INT TERM

free_port() {
  local port=$1
  local pids
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "  Port $port busy (pids: $pids) — killing"
    echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
}

wait_for_port() {
  local port=$1
  local name=$2
  local max=${3:-60}
  local i=0
  while ! lsof -ti tcp:"$port" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge "$max" ]; then
      echo "  $name did not open port $port within ${max}s"
      return 1
    fi
    sleep 1
  done
}

KEEP_TEMPORAL=0
if lsof -ti tcp:7233 >/dev/null 2>&1; then
  read -r -p "Temporal dev server is already running. Keep using it? [Y/n] " reply </dev/tty || reply=""
  case "$reply" in
    [nN]|[nN][oO]) KEEP_TEMPORAL=0 ;;
    *) KEEP_TEMPORAL=1 ;;
  esac
fi

echo "==> Freeing ports"
free_port "$API_PORT"   # API bridge
free_port 8234          # temporal UI proxy (iframe-friendly)
free_port 5173          # SPA dev server
if [ "$KEEP_TEMPORAL" -eq 0 ]; then
  free_port 7233        # temporal gRPC
  free_port 8233        # temporal UI
fi

if [ "$KEEP_TEMPORAL" -eq 1 ]; then
  existing_pid=$(lsof -ti tcp:7233 2>/dev/null | head -n1)
  echo "==> Keeping existing Temporal dev server on :7233 (pid: ${existing_pid:-unknown})"
else
  echo "==> Starting Temporal dev server (detached — survives Ctrl-C)"
  nohup temporal server start-dev >"$LOG_DIR/temporal.log" 2>&1 &
  temporal_pid=$!
  disown "$temporal_pid" 2>/dev/null || true
  echo "    pid: $temporal_pid"
  wait_for_port 7233 "Temporal" 30
fi

echo "==> Installing SPA deps (if needed)"
( cd dashboard/web && npm install --silent )

echo "==> Starting API bridge (auto-spawns a worker + UI proxy on :8234)"
uv run uvicorn dashboard.api.main:app --port "$API_PORT" >"$LOG_DIR/api.log" 2>&1 &
api_pid=$!
PIDS+=("$api_pid")
echo "    pid: $api_pid"
wait_for_port "$API_PORT" "API bridge" 30

echo "==> Starting SPA dev server"
( cd dashboard/web && npm run dev ) >"$LOG_DIR/web.log" 2>&1 &
web_pid=$!
PIDS+=("$web_pid")
echo "    pid: $web_pid"
wait_for_port 5173 "SPA dev server" 30

echo ""
echo "All services up:"
echo "  Dashboard:    http://localhost:5173"
echo "  API bridge:   http://localhost:$API_PORT"
echo "  Temporal UI:  http://localhost:8233"
echo "  Logs:         $LOG_DIR/"
echo ""
echo "Press Ctrl-C to stop everything (Temporal dev server stays up)."

wait
