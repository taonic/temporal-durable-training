# Durable LLM/Model Training on Temporal — Specification

> A demo that showcases Temporal as the orchestration layer for ML training:
> fault-tolerant, GPU-efficient, cyclic, and observable — without a black-box ML platform.

## 1. Goals (mapped to the agenda)

| # | Agenda goal | How this demo proves it |
|---|-------------|-------------------------|
| 1 | **Improve GPU utilization, research efficiency, model accuracy** | A Temporal-native **GPU resource pool** leases scarce GPUs to many jobs; a hyperparameter **sweep** keeps every GPU busy and surfaces the best model automatically. |
| 2 | **Build complex pipelines with higher durability than black-box alternatives** | The training loop is *your* code (epochs, early stopping, checkpoint policy) — not a vendor DAG. Every step is durable and resumable. |
| 3 | **Increase ML pipeline stability, repeatability, visibility for free** | Temporal UI is the experiment dashboard: every run, signal, retry, and metric is queryable. Deterministic replay = repeatability. No extra infra. |
| 4 | **Create cyclic, reusable ML Ops pipelines faster than state machines or DAGs** | Training is inherently cyclic (epoch loop, early-stop, retrain-on-drift). Workflows express loops/conditionals natively; DAGs can't. Sweep = reusable parent workflow. |
| ★ | **Human-in-the-loop (cross-cutting)** | A workflow pauses durably for a human — model-promotion approval gate, divergence intervention — for minutes or months, surviving restarts. DAGs/state machines can't pause for a human and resume in place. |

**Primary showcase:** *fault tolerance* (kill a worker / GPU mid-epoch → resume from last checkpoint, zero lost progress) and *GPU utilization* (N GPUs, M ≫ N jobs, pool keeps them saturated).

## 2. Non-goals

- Not a SOTA model or real LLM pretraining. The *orchestration* is the product, not the model.
- No distributed multi-node data-parallel training (single-process per job). Mentioned as a future extension.
- No cloud deployment in v1 — runs locally against `temporal server start-dev`.

## 3. What we borrow vs. what we improve (vs. `samingbar/temporal-ml-ops-samples`)

**Borrow:**
- Clean separation: deterministic **workflows** orchestrate; non-deterministic **activities** do ML.
- Checkpoint-aware resume: on retry, prefer an explicit checkpoint path, else auto-detect latest in `runs/{run_id}/`.
- Heartbeating from a long activity via a background thread (`asyncio.to_thread`) so Temporal sees liveness and can cancel.
- Pydantic models for all workflow/activity payloads. Stable `run_id` as the glue between training and serving.

**Improve:**
- **Epoch-level orchestration.** Reference runs all training in one 2-hour activity. We make **each epoch its own activity**, so checkpoints are first-class *workflow* state, retries are per-epoch, and the Temporal UI shows progress epoch-by-epoch. Workflow owns early-stopping and best-model logic.
- **GPU resource pool.** Reference ignores GPUs. We add a long-running pool workflow that leases a fixed set of GPUs to jobs (the showcase for goal 1).
- **Hyperparameter sweep.** A parent workflow fans out many child training runs over a search space, each leasing a GPU, collecting metrics, and picking a winner (goals 1 & 4).
- **Hybrid training backend.** Real PyTorch path *and* a deterministic simulator behind one interface, selected by config — runs on any laptop with no GPU, or on real hardware.
- **Built-in chaos.** A config-driven failure injector (crash mid-epoch, GPU "fault") to make fault tolerance demonstrable on demand.

## 4. Architecture

```
                         ┌──────────────────────────────┐
                         │  GpuPoolWorkflow (long-lived) │  ← models scarce GPUs
                         │  leases via Update/Signal     │
                         └──────────────┬────────────────┘
                                        │ lease / release (Update)
   SweepWorkflow (parent) ──fan out──►  TrainingWorkflow (child) × M
   - builds search space               - acquires GPU lease
   - starts N children at a time        - data prep activity
   - collects metrics                   - FOR epoch in range:           ◄── cyclic loop
   - picks best, registers model            train_epoch activity (heartbeats, checkpoints)
                                             evaluate activity
                                             early-stop / best-model decision
                                         - continue-as-new every K epochs (bounded history)
                                         - release GPU lease (finally)
                                         - register_model activity

   Worker(s) ── run on task queues ── activities do real/simulated ML + I/O
```

### 4.1 Workflows

- **`TrainingWorkflow`** — one model training run.
  - Input: `TrainingConfig` (dataset, hyperparams, max_epochs, checkpoint_every, early_stop_patience, backend, chaos).
  - Acquires a GPU lease from `GpuPoolWorkflow` (Update call) before training; releases in `finally`.
  - Loop over epochs; each epoch = `train_epoch` activity (resumes from `latest_checkpoint`), then `evaluate` activity.
  - Tracks `latest_checkpoint`, `best_metric`, `epochs_without_improvement` as workflow state.
  - Early stopping when patience exceeded. **Continue-as-new** every K epochs to keep event history small for long runs.
  - **Queries:** `progress()` → current epoch, best metric, latest checkpoint. **Signals:** `stop()` (graceful early stop), `update_lr()` (live hyperparam nudge — shows interactivity DAGs lack).
  - Output: `TrainingResult` (run_id, best_checkpoint, best_metric, history).

- **`GpuPoolWorkflow`** — durable semaphore over `num_gpus`.
  - State: list of GPU slots, each free or leased to a `run_id`.
  - **Update `acquire(run_id)`** → blocks (via `workflow.wait_condition`) until a slot is free, returns `gpu_id`.
  - **Update/Signal `release(gpu_id)`** → frees the slot.
  - **Query `utilization()`** → busy/total + current holders (drives the "GPU utilization" story in the UI).
  - Runs continuously; continue-as-new periodically.

- **`SweepWorkflow`** — hyperparameter search / reusable pipeline.
  - Input: `SweepConfig` (base config + search space grid/random, max_concurrency).
  - Starts child `TrainingWorkflow`s; concurrency is naturally bounded by the GPU pool (extra jobs wait in `acquire`).
  - Collects `TrainingResult`s, selects best, then enters a **human-approval gate** (§4.4b): waits on `approve`/`reject` signal (with a durable review-timeout) before calling `register_model`.
  - Demonstrates a *cyclic, reusable* pipeline: could loop to do successive-halving / retrain-on-new-data.

### 4.2 Activities (non-deterministic; the only place ML/IO happens)

- `prepare_data(cfg)` — load/snapshot dataset to `runs/{run_id}/data` (deterministic snapshot for repeatability).
- `train_epoch(cfg, epoch, resume_from)` — train exactly one epoch on leased GPU; **heartbeats** every few seconds from a background thread; honors cancellation; saves `checkpoint-{epoch}`; returns metrics + checkpoint path. Auto-detects latest checkpoint if `resume_from` is None (crash recovery). Injects chaos failures if configured.
- `evaluate(cfg, checkpoint)` — eval/val metrics.
- `register_model(run_id, checkpoint, metric)` — copy best checkpoint to `models/{name}` + write a model-card JSON (the "model registry").
- All payloads are Pydantic models; activities have explicit timeouts + retry policies.

### 4.3 Training backend (hybrid)

`backend: "sim" | "torch"` in config, behind a `TrainerBackend` protocol:
- **`sim`** (default): deterministic synthetic loss/accuracy curves seeded by hyperparams; sleeps a configurable per-step time to mimic GPU work; writes tiny JSON "checkpoints". Runs anywhere, fast, reproducible — ideal for live demos and CI.
- **`torch`**: real tiny model (small CNN on a downloaded image dataset, or a small HF transformer fine-tune) on CPU/GPU; real `.pt` checkpoints. Same interface, swapped by config.

The simulator and torch backend expose identical `train_one_epoch(state, resume_from) -> (metrics, checkpoint_path)` so workflow/activity code is backend-agnostic.

### 4.4 Chaos / fault injection

`ChaosConfig`: `crash_on_epoch` (raise mid-epoch to simulate worker death), `gpu_fault_rate`, `fail_first_n_attempts`. Combined with a documented "kill the worker with Ctrl-C" demo step, this makes fault tolerance reproducible. Temporal's activity retry + checkpoint resume = no lost epochs.

### 4.4b Human-in-the-loop (HITL)

ML training is rarely fully autonomous — humans gate promotions and intervene on bad runs. Temporal's killer capability here: a workflow can **block durably on a human decision for arbitrarily long** (`workflow.wait_condition` on a signal, backed by a durable timer) and survive worker/process restarts the entire time. A DAG/state machine can't pause for a human for three days and resume exactly where it left off. Two scenarios:

- **Primary — Model-promotion approval gate (in `SweepWorkflow`).** After the sweep selects the best run, the workflow enters a `PENDING_APPROVAL` state and waits for a human decision before promoting (registering/deploying) the model.
  - **Signal `approve(reviewer, note)` / `reject(reviewer, note)`** → unblocks the gate. **Query `pending_approval()`** → the candidate model, its metrics, and who/when, so the dashboard can render an approval card.
  - A configurable **review-timeout** (durable timer) escalates or auto-rejects if no human responds, so the demo never hangs forever. The reviewer identity + note are recorded in workflow history → auditable governance for free (goal 3).
  - This is the standard Temporal "human task" / async-approval pattern applied to MLOps.
- **Secondary — Divergence intervention (in `TrainingWorkflow`).** If `evaluate` reports a diverging/NaN/plateaued metric, the workflow pauses and waits for a human signal: **`resume_decision("continue" | "stop" | "adjust_lr", value)`**. Reuses the existing `update_lr`/`stop` machinery. Query `needs_attention()` drives a "this run needs you" badge in the dashboard.

Both decisions are issued from the dashboard (buttons → `POST /api/.../signal`) and are equally visible/replayable in the Temporal Web UI.

### 4.5 Dashboard UI (polished SPA)

A real-time web dashboard — the "wow" surface for the customer demo. **Design principle: every panel is visibly powered by Temporal** (Queries + Visibility API), and the dashboard is shown *side-by-side* with the Temporal Web UI so the audience sees the pretty view *and* the durable engine driving it. The SPA never holds authoritative state — Temporal does.

- **Backend bridge — `dashboard/api/` (FastAPI):** a thin read-mostly service that connects to Temporal and exposes:
  - `GET /api/runs` → list training workflows via the Visibility API (`list_workflows`), with status, type, run_id.
  - `GET /api/runs/{id}/progress` → calls the workflow's `progress()` **Query** (epoch, best metric, loss/acc history, latest checkpoint, retry count).
  - `GET /api/gpu` → calls `GpuPoolWorkflow.utilization()` **Query** (slots, holders, queue depth).
  - `GET /api/sweep/{id}` → child results + leaderboard.
  - `WS /ws` → pushes the above on a short poll interval so the UI animates live. (Polling Temporal queries, not a separate event bus — keeps Temporal as source of truth.)
  - A `POST /api/runs` + `POST /api/runs/{id}/signal` (stop / update_lr) so the demo can *drive* Temporal from the UI (start a run, nudge LR live, trigger chaos).
- **Frontend — `dashboard/web/` (React + Vite + TypeScript + Tailwind, charts via Recharts):**
  - **GPU Pool panel:** a grid of GPU cards that light up as leased (run_id, elapsed), with a queue-depth badge — the visual centerpiece for goal 1. Watch 2 GPUs stay saturated while 4 jobs wait.
  - **Live training panel:** loss & accuracy curves per run, current epoch, best metric, and a **retry/recovery banner** that flashes when an epoch is retried after a crash — the fault-tolerance money shot.
  - **Approval card (HITL):** when a sweep is `PENDING_APPROVAL`, a prominent card shows the winning model + metrics with **Approve / Reject** buttons (→ signal the workflow) and a live countdown to the review-timeout. A "needs attention" badge appears for diverging runs with continue/stop/adjust-LR controls.
  - **Sweep leaderboard:** sortable table of configs → metrics, winner highlighted, link to registered model card.
  - **Run timeline:** compact list of all runs with status badges; click-through opens the *Temporal UI* for that workflow (explicit "this is running on Temporal" link).
  - Tasteful animation/transitions (Framer Motion) for the demo polish requested.
- **One-command launch:** `dashboard` script runs API + Vite dev server (or serves a built bundle) so the presenter starts it in one step.

> Trade-off acknowledged: the SPA is the largest single piece of build/maintenance here and could distract from Temporal's own UI. Mitigated by (a) keeping it read-mostly over Temporal Queries, (b) the side-by-side presentation, and (c) click-through links into the Temporal UI from every run.

## 5. Project layout

```
temporal-durable-training/
├── SPEC.md                     # this file
├── README.md                   # quickstart + the three demo scripts
├── pyproject.toml              # deps: temporalio, pydantic; optional [torch] extra
├── src/durable_training/
│   ├── shared.py               # Pydantic models (configs, results, checkpoint meta)
│   ├── workflows/
│   │   ├── training.py         # TrainingWorkflow
│   │   ├── gpu_pool.py         # GpuPoolWorkflow
│   │   └── sweep.py            # SweepWorkflow
│   ├── activities/
│   │   ├── training.py         # prepare_data, train_epoch, evaluate, register_model
│   │   └── backends/
│   │       ├── base.py         # TrainerBackend protocol
│   │       ├── simulator.py    # deterministic sim backend
│   │       └── torch_backend.py# real PyTorch backend (optional extra)
│   ├── worker.py               # registers everything on task queues
│   └── chaos.py                # failure injection helpers
├── dashboard/
│   ├── api/                    # FastAPI bridge → Temporal Queries + Visibility API
│   │   └── main.py
│   └── web/                    # React + Vite + TS + Tailwind SPA
│       ├── src/                # GPU grid, live curves, sweep leaderboard, run timeline
│       └── package.json
├── scripts/
│   ├── demo_fault_tolerance.py # start a run, instruct to kill worker, show resume
│   ├── demo_gpu_pool.py        # 2 GPUs, 6 jobs → watch pool saturate
│   ├── demo_sweep.py           # hyperparameter sweep → best model
│   └── demo_human_in_the_loop.py # sweep → approval gate → approve/reject from CLI or UI
└── tests/
    ├── test_training_workflow.py  # time-skipping env: checkpoint resume, early stop
    ├── test_gpu_pool.py           # lease/release, blocking acquire
    └── test_sweep.py              # picks best result
```

## 6. Demo scripts (the live narrative)

1. **Fault tolerance** (`demo_fault_tolerance.py`)
   - Start a `TrainingWorkflow` (10 epochs). Mid-run, **kill the worker** (or set `crash_on_epoch=4`).
   - Restart the worker → `train_epoch` retries, auto-detects `checkpoint-3`, resumes at epoch 4. UI shows the retry and unbroken progress. **Zero lost epochs.**

2. **GPU utilization** (`demo_gpu_pool.py`)
   - `GpuPoolWorkflow` with `num_gpus=2`. Launch **6** training jobs.
   - Query `utilization()` → 2 busy, 4 queued; as jobs finish, queued ones acquire immediately. GPUs stay saturated; no manual scheduling code.

3. **Cyclic reusable pipeline** (`demo_sweep.py`)
   - `SweepWorkflow` over a small grid (e.g., 3 learning rates × 2 batch sizes = 6 runs).
   - Pool bounds concurrency to GPU count; sweep collects metrics, picks the best, registers the model. One reusable parent workflow = an entire experiment.

4. **Human-in-the-loop approval** (`demo_human_in_the_loop.py`)
   - Run a sweep; when it finishes it parks in `PENDING_APPROVAL` and **waits** (durably) for a human.
   - The presenter clicks **Approve** (or **Reject**) in the dashboard — or runs the CLI signal — and the model is (or isn't) promoted. Optionally kill the worker *while it's waiting* to prove the pause survives restarts. The reviewer + note land in workflow history (auditable).

All three are observable live in **two** places, shown side-by-side: the **custom dashboard** (GPU grid, live curves, leaderboard, retry banner) and the **Temporal Web UI** (`localhost:8233` — histories, signals, queries, retries, child workflows). The dashboard's panels are driven by Temporal Queries/Visibility, so the audience sees that the engine *is* Temporal.

## 7. Determinism & correctness rules (Temporal)

- Workflows: no I/O, no clocks, no RNG, no threads. All randomness/time/data access happens in activities. Hyperparameter sampling for the sweep is done deterministically (seeded) or in an activity.
- Long loops use **continue-as-new** to bound history.
- Activities are idempotent w.r.t. `run_id` + epoch (re-running an epoch overwrites its checkpoint safely).
- Explicit timeouts: `start_to_close` per activity; `heartbeat_timeout` for `train_epoch`. Retry policies with backoff; checkpoint resume makes retries cheap.

## 8. Tech / versions

- **Backend:** Python 3.10+; `temporalio` (latest), `pydantic` v2; `fastapi` + `uvicorn` for the dashboard API.
- Optional `torch` extra for the real backend. Default `sim` backend has no heavy deps.
- **Frontend:** React 18 + Vite + TypeScript + Tailwind; Recharts (charts), Framer Motion (animation). Dev via Vite, build to a static bundle the API can serve.
- Local Temporal via `temporal server start-dev` (UI at :8233).
- Tests use `temporalio.testing.WorkflowEnvironment` time-skipping (fast, no server needed).

## 9. Acceptance criteria

- [ ] `sim` backend trains end-to-end with no GPU and no torch installed.
- [ ] Killing the worker mid-training resumes from the last checkpoint with no lost epochs (demo 1).
- [ ] With `num_gpus=2` and 6 jobs, at most 2 run concurrently and all complete; `utilization()` reflects it (demo 2).
- [ ] Sweep completes all configs and registers the best model (demo 3).
- [ ] Workflow tests pass under the time-skipping test environment, including a forced mid-epoch crash + resume.
- [ ] `torch` backend trains a tiny real model when the extra is installed.
- [ ] Dashboard SPA renders the GPU grid, live loss/acc curves, and sweep leaderboard, all fed by Temporal Queries/Visibility; the retry banner fires on a mid-epoch crash; each run links into the Temporal Web UI.
- [ ] HITL: a finished sweep parks in `PENDING_APPROVAL`, survives a worker restart while waiting, and promotes the model only after an Approve signal (from UI or CLI); Reject skips promotion; the review-timeout auto-resolves. Reviewer + note are recorded in history.

## 10. Open questions / future work

- Real distributed (multi-GPU/multi-node) training as activities coordinated by a workflow.
- Retrain-on-drift loop (a cron/scheduled workflow) to fully exploit the "cyclic" story.
- Swap the registry activity for a real model registry (MLflow/W&B) — trivial since it's just an activity.
- **Active-learning HITL loop** (scenario #3): model surfaces uncertain samples → human labels → training resumes. Deferred from v1 as the heaviest HITL variant.
```
