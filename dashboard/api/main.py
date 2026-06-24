"""Dashboard API bridge.

A thin FastAPI service that turns Temporal Queries + the Visibility API into JSON
for the SPA. It holds NO authoritative state — Temporal does. Every panel in the
UI is therefore provably driven by Temporal: the GPU grid is a pool Query, the
training curves are a workflow Query, the run list is a Visibility list.

Run:
  uv sync --extra dashboard
  uv run python -m durable_training.worker          # in another terminal
  uv run uvicorn dashboard.api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import socket
import asyncio
import contextlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from temporalio.api.enums.v1 import TaskQueueType
from temporalio.api.taskqueue.v1 import TaskQueue
from temporalio.api.workflowservice.v1 import DescribeTaskQueueRequest

from dashboard.api.ui_proxy import PROXY_HOST, PROXY_PORT
from dashboard.api.ui_proxy import app as ui_proxy_app
from durable_training.common import connect, ensure_gpu_pool
from durable_training.shared import (
    GPU_POOL_WORKFLOW_ID,
    TASK_QUEUE,
    SweepConfig,
    TrainingConfig,
    TrainingInput,
)

# Worker processes this API has spawned (so it can list + stop them).
_spawned: dict[int, subprocess.Popen] = {}

_HOST = socket.gethostname()


def _api_port(default: int = 8000) -> int:
    """The port uvicorn was launched with (parsed from argv), for the banner."""
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            with contextlib.suppress(ValueError):
                return int(argv[i + 1])
        if a.startswith("--port="):
            with contextlib.suppress(ValueError):
                return int(a.split("=", 1)[1])
    return default


def _alive_pids(pids: list[int]) -> set[int]:
    """Ask the OS (`ps`) which of these PIDs are actually still running.

    Temporal's poller info lags ~a minute behind a dead worker, so we verify
    liveness directly against the process table (one `ps` for all pids)."""
    if not pids:
        return set()
    try:
        r = subprocess.run(
            ["ps", "-o", "pid=,stat=", "-p", ",".join(str(p) for p in pids)],
            capture_output=True,
            text=True,
        )
    except Exception:
        return set(pids)  # can't check — assume alive rather than hide them
    alive: set[int] = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        # A killed-but-unreaped child shows as a zombie (stat 'Z'); treat as dead.
        if parts[1].startswith("Z"):
            continue
        try:
            alive.add(int(parts[0]))
        except ValueError:
            pass
    return alive

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run the header-stripping Temporal UI proxy in-process on its own port, so a
    single `uvicorn dashboard.api.main:app` serves both the API and the embeddable
    UI. (It needs a separate port: the UI loads root-absolute assets that would
    collide with this app's SPA mount on `/`.)"""
    proxy = uvicorn.Server(
        # log_config=None: don't let the nested server reconfigure global logging
        # (it would suppress the main API server's own startup/INFO lines).
        uvicorn.Config(
            ui_proxy_app,
            host=PROXY_HOST,
            port=PROXY_PORT,
            log_level="warning",
            log_config=None,
        )
    )
    task = asyncio.create_task(proxy.serve())

    # Start one worker so the pool isn't empty when the dashboard opens — the
    # first "Create job" then has somewhere to run.
    with contextlib.suppress(Exception):
        proc = subprocess.Popen([sys.executable, "-m", "durable_training.worker"])
        _spawned[proc.pid] = proc

    port = _api_port()
    if _dist.exists():
        # Built SPA is mounted on this app — the dashboard is served here.
        print(f"\n  ✓ Dashboard ready → http://localhost:{port}\n", flush=True)
    else:
        # Dev flow: the React UI runs on Vite; this process is just the API.
        print(
            f"\n  ✓ API ready → http://localhost:{port}"
            "\n    Open the dashboard at → http://localhost:5173"
            " (run `npm run dev` in dashboard/web)\n",
            flush=True,
        )

    try:
        yield
    finally:
        proxy.should_exit = True
        for p in list(_spawned.values()):
            with contextlib.suppress(Exception):
                p.terminate()
        with contextlib.suppress(Exception):
            await task


app = FastAPI(title="Durable Training Dashboard API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_client = None
TEMPORAL_UI = "http://localhost:8233"
# The browser-reachable URL of the header-stripping UI proxy. Locally it's
# localhost:8234; in a Daytona sandbox the launcher injects the signed preview URL.
TEMPORAL_UI_PROXY = os.environ.get("TEMPORAL_UI_PROXY_URL", "http://localhost:8234")


async def client():
    global _client
    if _client is None:
        _client = await connect()
        await ensure_gpu_pool(_client, num_gpus=2)
    return _client


async def _query(workflow_id: str, name: str) -> Optional[Any]:
    """Query a workflow by id; returns the decoded value (dict) or None."""
    try:
        handle = (await client()).get_workflow_handle(workflow_id)
        return await handle.query(name)
    except Exception:
        return None


async def _retry_info(workflow_id: str) -> dict:
    """Surface activity retry state + which worker is running it (recovery banner
    and the 'job is on worker X' label)."""
    try:
        desc = await (await client()).get_workflow_handle(workflow_id).describe()
        pending = list(desc.raw_description.pending_activities)
        attempt = max((pa.attempt for pa in pending), default=1)
        failing = [pa for pa in pending if pa.HasField("last_failure")]
        worker = pending[0].last_worker_identity if pending else None
        # last_worker_identity lingers on a killed worker until the activity is
        # rescheduled — don't show a dead worker as the one running the job.
        if worker and "@" in worker:
            try:
                pid, host = worker.split("@", 1)
                if host == _HOST and int(pid) not in _alive_pids([int(pid)]):
                    worker = None
            except ValueError:
                pass
        return {
            "retrying": attempt > 1 or bool(failing),
            "retry_attempt": attempt,
            "last_failure": failing[0].last_failure.message if failing else "",
            "worker": worker or None,
        }
    except Exception:
        return {"retrying": False, "retry_attempt": 1, "last_failure": "", "worker": None}


async def _list(workflow_type: str) -> list[dict]:
    try:
        c = await client()
    except Exception:
        return []  # Temporal not reachable yet
    out: list[dict] = []
    async for wf in c.list_workflows(f"WorkflowType = '{workflow_type}'"):
        out.append(
            {
                "id": wf.id,
                "run_id": wf.run_id,
                "status": wf.status.name if wf.status else "UNKNOWN",
                "temporal_url": f"{TEMPORAL_UI}/namespaces/default/workflows/{wf.id}",
            }
        )
    return out


async def _workers() -> list[dict]:
    """List workers currently polling the training task queue (the worker pool).

    Uses describe_task_queue's poller info — so it reflects *every* worker on the
    queue (started via the launcher, the CLI, or this dashboard), keyed by the
    Temporal identity (``<pid>@<host>``)."""
    try:
        c = await client()
    except Exception:
        return []  # Temporal not reachable yet
    out: dict[str, dict] = {}
    for qtype, key in (
        (TaskQueueType.TASK_QUEUE_TYPE_WORKFLOW, "workflow"),
        (TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY, "activity"),
    ):
        try:
            resp = await c.workflow_service.describe_task_queue(
                DescribeTaskQueueRequest(
                    namespace="default",
                    task_queue=TaskQueue(name=TASK_QUEUE),
                    task_queue_type=qtype,
                )
            )
        except Exception:
            continue
        for p in resp.pollers:
            e = out.setdefault(
                p.identity,
                {"identity": p.identity, "workflow": False, "activity": False},
            )
            e[key] = True
            try:
                e["last_access"] = p.last_access_time.ToDatetime().isoformat()
            except Exception:
                e["last_access"] = None
            try:
                pid = int(p.identity.split("@")[0])
            except ValueError:
                pid = None
            e["pid"] = pid
            e["host"] = p.identity.split("@", 1)[1] if "@" in p.identity else ""
            e["spawned"] = pid in _spawned

    # Verify liveness via the OS for workers on this host; remote hosts are
    # left as None (can't be `ps`-checked from here).
    local = [e["pid"] for e in out.values() if e["pid"] and e.get("host") == _HOST]
    alive = _alive_pids(local)
    for e in out.values():
        if e["pid"] and e.get("host") == _HOST:
            e["alive"] = e["pid"] in alive
        else:
            e["alive"] = None
    return list(out.values())


# --- read endpoints (Temporal Queries / Visibility) -------------------------


@app.get("/api/gpu")
async def gpu():
    return await _query(GPU_POOL_WORKFLOW_ID, "utilization") or {
        "total": 0, "busy": 0, "free": 0, "queue_depth": 0, "slots": [], "waiting": []
    }


@app.get("/api/runs")
async def runs():
    return await _list("TrainingWorkflow")


@app.get("/api/runs/{run_id}/progress")
async def run_progress(run_id: str):
    return await _query(run_id, "progress") or {}


@app.get("/api/sweeps")
async def sweeps():
    return await _list("SweepWorkflow")


@app.get("/api/workers")
async def workers():
    return await _workers()


@app.post("/api/workers")
async def add_worker():
    """Spawn a new worker process on this host, joining the task queue pool."""
    proc = subprocess.Popen([sys.executable, "-m", "durable_training.worker"])
    _spawned[proc.pid] = proc
    return {"pid": proc.pid}


@app.delete("/api/workers/{pid}")
async def kill_worker(pid: int):
    """Stop a worker this dashboard spawned (simulates a node going away)."""
    proc = _spawned.pop(pid, None)
    if proc is None:
        return {"ok": False, "error": "not a dashboard-spawned worker"}
    proc.kill()
    # Reap it (we're its parent) so it doesn't linger as a zombie that `ps` — and
    # therefore the worker-pool liveness check — would still report as alive.
    with contextlib.suppress(Exception):
        await asyncio.to_thread(proc.wait, 5)
    return {"ok": True, "killed": pid}


@app.get("/api/sweeps/{sweep_id}")
async def sweep_status(sweep_id: str):
    return await _query(sweep_id, "status") or {}


# --- write endpoints (start workflows, signal humans-in-the-loop) -----------


@app.post("/api/runs")
async def start_run(cfg: TrainingConfig):
    c = await client()
    await c.start_workflow(
        "TrainingWorkflow",
        TrainingInput(config=cfg),
        id=cfg.run_id,
        task_queue=TASK_QUEUE,
    )
    return {"run_id": cfg.run_id}


@app.post("/api/sweeps")
async def start_sweep(cfg: SweepConfig):
    c = await client()
    await c.start_workflow("SweepWorkflow", cfg, id=cfg.name, task_queue=TASK_QUEUE)
    return {"name": cfg.name}


class ResizeBody(BaseModel):
    num_gpus: int


@app.post("/api/gpu/resize")
async def resize_gpu(body: ResizeBody):
    c = await client()
    # Start a fresh pool sized to the request if the pool workflow isn't running
    # (e.g. it completed or the dev server was restarted); no-op if it's running.
    await ensure_gpu_pool(c, num_gpus=body.num_gpus)
    try:
        await c.get_workflow_handle(GPU_POOL_WORKFLOW_ID).signal(
            "resize", args=[body.num_gpus]
        )
    except Exception:
        pass  # just-created pool is already at the requested size
    return {"ok": True, "num_gpus": body.num_gpus}


@app.post("/api/gpu/{gpu_id}/kill")
async def kill_gpu(gpu_id: int):
    """Fail a GPU. If a job holds it, that job is revoked and re-acquires another."""
    handle = (await client()).get_workflow_handle(GPU_POOL_WORKFLOW_ID)
    await handle.signal("kill_gpu", args=[gpu_id])
    return {"ok": True, "killed_gpu": gpu_id}


class SignalBody(BaseModel):
    signal: str
    args: list[Any] = []


@app.post("/api/runs/{run_id}/signal")
async def signal_run(run_id: str, body: SignalBody):
    handle = (await client()).get_workflow_handle(run_id)
    # args= (not *args) so multi-argument signals (e.g. resume_decision) work.
    await handle.signal(body.signal, args=body.args)
    return {"ok": True}


class DecisionBody(BaseModel):
    reviewer: str = "dashboard-user"
    note: str = ""


@app.post("/api/sweeps/{sweep_id}/approve")
async def approve_sweep(sweep_id: str, body: DecisionBody):
    handle = (await client()).get_workflow_handle(sweep_id)
    await handle.signal("approve", args=[body.reviewer, body.note])
    return {"ok": True}


@app.post("/api/sweeps/{sweep_id}/reject")
async def reject_sweep(sweep_id: str, body: DecisionBody):
    handle = (await client()).get_workflow_handle(sweep_id)
    await handle.signal("reject", args=[body.reviewer, body.note])
    return {"ok": True}


# --- websocket: push aggregated state so the UI animates live ---------------


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                run_list = await _list("TrainingWorkflow")
                sweep_list = await _list("SweepWorkflow")
                progresses = await asyncio.gather(
                    *[_query(r["id"], "progress") for r in run_list]
                )
                retries = await asyncio.gather(*[_retry_info(r["id"]) for r in run_list])
                sweep_statuses = await asyncio.gather(
                    *[_query(s["id"], "status") for s in sweep_list]
                )
                runs_payload = []
                for r, p, retry in zip(run_list, progresses, retries):
                    if not p:
                        continue
                    runs_payload.append({**p, **retry, "temporal_url": r["temporal_url"]})
                payload = {
                    "gpu": await _query(GPU_POOL_WORKFLOW_ID, "utilization"),
                    "runs": runs_payload,
                    "sweeps": [s for s in sweep_statuses if s],
                    "workers": await _workers(),
                    "temporal_ui": TEMPORAL_UI,
                    "temporal_ui_proxy": TEMPORAL_UI_PROXY,
                }
            except Exception:
                # Temporal not reachable yet — degrade to an empty frame and retry,
                # so the dashboard shows defaults instead of the socket erroring out.
                payload = {
                    "gpu": None,
                    "runs": [],
                    "sweeps": [],
                    "workers": [],
                    "temporal_ui": TEMPORAL_UI,
                    "temporal_ui_proxy": TEMPORAL_UI_PROXY,
                    "temporal_unavailable": True,
                }
            await websocket.send_json(payload)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


# --- serve the built SPA if present -----------------------------------------

_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="spa")
