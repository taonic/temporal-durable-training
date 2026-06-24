import { useEffect, useRef, useState } from "react";
import type { DashboardState } from "./types";

/** Live dashboard state via the API's websocket (which polls Temporal Queries). */
export function useDashboard(): { state: DashboardState; connected: boolean } {
  const [state, setState] = useState<DashboardState>({
    gpu: null,
    runs: [],
    sweeps: [],
    workers: [],
    temporal_ui: "http://localhost:8233",
    temporal_ui_proxy: "http://localhost:8234",
  });
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let stop = false;
    function connect() {
      if (stop) return;
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onmessage = (e) => setState(JSON.parse(e.data));
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
    }
    connect();
    return () => {
      stop = true;
      wsRef.current?.close();
    };
  }, []);

  return { state, connected };
}

export async function startSweep(name: string) {
  await fetch("/api/sweeps", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      name,
      base: { run_id: name, model_name: "dashboard-sweep", max_epochs: 4, steps_per_epoch: 6 },
      search_space: { learning_rate: [0.001, 0.01, 0.1], batch_size: [16, 32] },
      review_timeout_seconds: 900,
    }),
  });
}

export async function startRun(runId: string) {
  await fetch("/api/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      run_id: runId,
      model_name: "dashboard-run",
      max_epochs: 5,
      steps_per_epoch: 5, // ~5s per epoch at the default 1s/step
      register_on_complete: true,
    }),
  });
}

export async function decideSweep(sweepId: string, approve: boolean) {
  await fetch(`/api/sweeps/${sweepId}/${approve ? "approve" : "reject"}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ reviewer: "dashboard-user", note: approve ? "approved in UI" : "rejected in UI" }),
  });
}

export async function addWorker() {
  await fetch("/api/workers", { method: "POST" });
}

export async function killWorker(pid: number) {
  await fetch(`/api/workers/${pid}`, { method: "DELETE" });
}

export async function resizeGpu(numGpus: number) {
  await fetch("/api/gpu/resize", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ num_gpus: Math.max(1, numGpus) }),
  });
}

export async function killGpu(gpuId: number) {
  await fetch(`/api/gpu/${gpuId}/kill`, { method: "POST" });
}

export async function signalRun(runId: string, signal: string, args: unknown[] = []) {
  await fetch(`/api/runs/${runId}/signal`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ signal, args }),
  });
}
