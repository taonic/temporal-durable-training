"""TrainingWorkflow — one durable, resumable training run.

Orchestrates the epoch loop: acquire a GPU, then for each epoch run train +
evaluate activities, track the best checkpoint, early-stop, and (optionally)
pause for a human. Each completed epoch is checkpointed, so a worker crash costs
at most one epoch — the rest resumes in place. Long runs continue-as-new to keep
event history bounded.

NB: no ``from __future__ import annotations`` here — Temporal decodes workflow
arguments from the run method's real type hints; stringized annotations fail to
resolve at decode time and args would silently arrive as plain dicts.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from durable_training.activities.training import (
        evaluate,
        prepare_data,
        register_model,
        simulate_step,
        train_epoch,
    )
    from durable_training.shared import (
        AcquireGpuInput,
        AttentionRequest,
        EpochMetrics,
        EvaluateInput,
        JobStep,
        RegisterModelInput,
        ReleaseGpuInput,
        TrainEpochInput,
        TrainingConfig,
        TrainingInput,
        TrainingProgress,
        TrainingResult,
        TrainingResumeState,
    )

# The pipeline every training job moves through. (name, label, kind, rel. duration).
# "train" is the actual epoch loop and is always the longest step.
PIPELINE: list[tuple[str, str, str, float]] = [
    ("interpret", "Interpret training intent", "workflow", 0.4),
    ("build", "Build image and deps", "activity", 1.2),
    ("quota", "Query quota and cluster options", "activity", 0.6),
    ("choose", "Choose target cluster", "workflow", 0.4),
    ("reserve", "Reserve GPU", "activity", 1.0),
    ("spec", "Materialize job spec", "history", 0.3),
    ("train", "Submit and monitor with k8s", "activity", 0.0),
]


@workflow.defn
class TrainingWorkflow:
    def __init__(self) -> None:
        self._cfg: TrainingConfig | None = None
        self._status = "pending"
        self._current_epoch = 0
        self._best_metric: float | None = None
        self._best_epoch = -1
        self._best_checkpoint: str | None = None
        self._latest_checkpoint: str | None = None
        self._history: list[EpochMetrics] = []
        self._steps: list[JobStep] = []
        self._patience = 0
        self._gpu_id: int | None = None
        self._lease_seq = 0
        # interactivity / HITL
        self._stop = False
        self._lr_override: float | None = None
        self._attention: AttentionRequest | None = None
        self._decision: tuple[str, float] | None = None
        self._continuing = False

    # --- main loop ----------------------------------------------------------

    @workflow.run
    async def run(self, input: TrainingInput) -> TrainingResult:
        cfg = input.config
        self._cfg = cfg
        self._steps = [
            JobStep(name=n, label=lbl, kind=k) for (n, lbl, k, _) in PIPELINE
        ]
        start_epoch = 0
        if input.resume is not None:
            self._restore(input.resume)  # restores steps (preamble already done)
            start_epoch = input.resume.next_epoch

        try:
            # Steps 1–6 run once (skipped when resuming from continue-as-new).
            if input.resume is None:
                await self._run_preamble(cfg)

            self._set_step("train", "running")
            self._status = "training"
            for epoch in range(start_epoch, cfg.max_epochs):
                if self._stop:
                    break
                self._current_epoch = epoch

                tr = await self._run_epoch(cfg, epoch)
                ev = await workflow.execute_activity(
                    evaluate,
                    EvaluateInput(
                        config=cfg,
                        epoch=epoch,
                        checkpoint_path=tr.checkpoint_path,
                        gpu_id=self._gpu_id if self._gpu_id is not None else -1,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                )

                m = EpochMetrics(
                    epoch=epoch,
                    train_loss=tr.train_loss,
                    val_loss=ev.val_loss,
                    val_accuracy=ev.val_accuracy,
                    checkpoint_path=tr.checkpoint_path,
                )
                self._history.append(m)
                self._latest_checkpoint = tr.checkpoint_path

                # HITL: pause for a human at a configured epoch (intervention demo).
                if cfg.pause_for_review_on_epoch == epoch:
                    await self._await_human(m)
                    if self._stop:
                        break

                # Best-model tracking + early stopping.
                if self._best_metric is None or m.val_accuracy > self._best_metric:
                    self._best_metric = m.val_accuracy
                    self._best_epoch = epoch
                    self._best_checkpoint = m.checkpoint_path
                    self._patience = 0
                else:
                    self._patience += 1
                    if self._patience >= cfg.early_stop_patience:
                        workflow.logger.info("Early stopping at epoch %d", epoch)
                        break

                # Bound history for very long runs.
                if (
                    cfg.continue_as_new_every > 0
                    and (epoch + 1) % cfg.continue_as_new_every == 0
                    and (epoch + 1) < cfg.max_epochs
                ):
                    self._continuing = True
                    workflow.continue_as_new(
                        TrainingInput(config=cfg, resume=self._snapshot(epoch + 1))
                    )

            self._set_step("train", "done")
            self._status = "completed"
            if cfg.register_on_complete and self._best_checkpoint is not None:
                await workflow.execute_activity(
                    register_model,
                    RegisterModelInput(
                        run_id=cfg.run_id,
                        model_name=cfg.model_name,
                        checkpoint_path=self._best_checkpoint,
                        metric=self._best_metric or 0.0,
                        reviewer="auto",
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                )
            return self._result()
        finally:
            # Release the GPU on completion/failure — but NOT across continue-as-new,
            # where the successor run keeps the same lease.
            if not self._continuing and self._gpu_id is not None and cfg.require_gpu:
                await workflow.execute_activity(
                    "release_gpu",
                    ReleaseGpuInput(pool_id=cfg.gpu_pool_id, gpu_id=self._gpu_id),
                    start_to_close_timeout=timedelta(minutes=5),
                )
                # GPU released — clear it so the UI shows no GPU for the finished run.
                self._gpu_id = None

    # --- pipeline steps -----------------------------------------------------

    def _set_step(self, name: str, status: str) -> None:
        for s in self._steps:
            if s.name == name:
                s.status = status

    async def _run_preamble(self, cfg: TrainingConfig) -> None:
        """Steps 1–6: interpret intent, build image, query quota, choose cluster,
        reserve machines + locate data, materialize job spec. Each simulates work;
        the actual training (step 7) is the longest."""
        durations = {n: d * cfg.pipeline_seconds for (n, _, _, d) in PIPELINE}

        for name, label, kind, _ in PIPELINE:
            if name == "train":
                break
            self._set_step(name, "running")
            if name == "reserve":
                # Reserve resources: lease a GPU, then locate/prepare the dataset.
                if cfg.require_gpu and self._gpu_id is None:
                    await self._acquire_gpu(cfg)
                await workflow.execute_activity(
                    prepare_data, cfg, start_to_close_timeout=timedelta(minutes=5)
                )
                if durations[name] > 0:
                    await workflow.execute_activity(
                        simulate_step,
                        args=[label, durations[name]],
                        start_to_close_timeout=timedelta(minutes=5),
                        heartbeat_timeout=timedelta(seconds=30),
                    )
            elif kind == "activity" and durations[name] > 0:
                await workflow.execute_activity(
                    simulate_step,
                    args=[label, durations[name]],
                    start_to_close_timeout=timedelta(minutes=5),
                    heartbeat_timeout=timedelta(seconds=30),
                )
            elif durations[name] > 0:
                # workflow/history steps are pure orchestration — a durable timer.
                await workflow.sleep(timedelta(seconds=durations[name]))
            self._set_step(name, "done")

    async def _acquire_gpu(self, cfg: TrainingConfig) -> None:
        self._status = "acquiring_gpu"
        # The broker heartbeats while waiting in the GPU queue, so a short
        # heartbeat timeout keeps liveness tight; if the per-attempt window
        # elapses the activity retries and resumes the same (idempotent) lease.
        self._gpu_id = await workflow.execute_activity(
            "acquire_gpu",
            AcquireGpuInput(
                pool_id=cfg.gpu_pool_id, run_id=cfg.run_id, lease_seq=self._lease_seq
            ),
            start_to_close_timeout=timedelta(minutes=5),
            heartbeat_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=5),
            ),
        )

    async def _reacquire(self, cfg: TrainingConfig) -> None:
        if not cfg.require_gpu:
            return
        self._lease_seq += 1  # fresh lease id, so we don't get the dead GPU back
        await self._acquire_gpu(cfg)
        self._status = "training"

    async def _run_epoch(self, cfg: TrainingConfig, epoch: int):
        """Run one epoch. train_epoch pings the GPU as it trains; if that ping
        fails (GPU revoked) the activity fails with GpuFailure — we then drop back
        to Reserve GPU, lease a healthy one, and retry the epoch from checkpoint."""
        while True:
            gpu = self._gpu_id if self._gpu_id is not None else -1
            try:
                return await workflow.execute_activity(
                    train_epoch,
                    TrainEpochInput(
                        config=cfg,
                        epoch=epoch,
                        resume_from=self._latest_checkpoint,
                        gpu_id=gpu,
                        lr_override=self._lr_override,
                    ),
                    start_to_close_timeout=timedelta(minutes=30),
                    heartbeat_timeout=timedelta(seconds=cfg.heartbeat_timeout_seconds),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=1),
                        maximum_interval=timedelta(seconds=5),
                    ),
                    # User metadata shown in the Temporal UI for this activity.
                    summary=f"epoch {epoch} · GPU {gpu}",
                )
            except ActivityError as err:
                if not (
                    isinstance(err.cause, ApplicationError)
                    and err.cause.type == "GpuFailure"
                ):
                    raise
                # GPU revoked — drop back to Reserve GPU and lease a healthy one.
                self._gpu_id = None  # clear the dead GPU so the UI shows no GPU
                self._status = "gpu_revoked"
                self._set_step("train", "pending")
                self._set_step("reserve", "running")
                await self._reacquire(cfg)
                self._set_step("reserve", "done")
                self._set_step("train", "running")

    async def _await_human(self, m: EpochMetrics) -> None:
        self._attention = AttentionRequest(
            epoch=m.epoch, reason="manual review requested", val_loss=m.val_loss
        )
        self._status = "needs_attention"
        await workflow.wait_condition(lambda: self._decision is not None)
        action, value = self._decision  # type: ignore[misc]
        self._decision = None
        self._attention = None
        self._status = "training"
        if action == "stop":
            self._stop = True
        elif action == "adjust_lr":
            self._lr_override = value
        workflow.logger.info("Human decision at epoch %d: %s", m.epoch, action)

    # --- signals / queries --------------------------------------------------

    @workflow.signal
    def stop(self) -> None:
        self._stop = True

    @workflow.signal
    def update_lr(self, lr: float) -> None:
        self._lr_override = lr

    @workflow.signal
    def resume_decision(self, action: str, value: float = 0.0) -> None:
        self._decision = (action, value)

    @workflow.query
    def progress(self) -> TrainingProgress:
        return TrainingProgress(
            run_id=self._cfg.run_id if self._cfg else "",
            status=self._status,
            current_epoch=self._current_epoch,
            max_epochs=self._cfg.max_epochs if self._cfg else 0,
            best_metric=self._best_metric,
            best_epoch=self._best_epoch,
            latest_checkpoint=self._latest_checkpoint,
            gpu_id=self._gpu_id,
            history=self._history,
            steps=self._steps,
            needs_attention=self._attention,
        )

    # --- continue-as-new plumbing ------------------------------------------

    def _snapshot(self, next_epoch: int) -> TrainingResumeState:
        return TrainingResumeState(
            next_epoch=next_epoch,
            gpu_id=self._gpu_id,
            lease_seq=self._lease_seq,
            steps=self._steps,
            best_metric=self._best_metric,
            best_epoch=self._best_epoch,
            best_checkpoint=self._best_checkpoint,
            latest_checkpoint=self._latest_checkpoint,
            history=self._history,
            patience=self._patience,
        )

    def _restore(self, s: TrainingResumeState) -> None:
        self._gpu_id = s.gpu_id
        self._lease_seq = s.lease_seq
        if s.steps:
            self._steps = list(s.steps)
        self._best_metric = s.best_metric
        self._best_epoch = s.best_epoch
        self._best_checkpoint = s.best_checkpoint
        self._latest_checkpoint = s.latest_checkpoint
        self._history = list(s.history)
        self._patience = s.patience

    def _result(self) -> TrainingResult:
        return TrainingResult(
            run_id=self._cfg.run_id if self._cfg else "",
            status=self._status,
            best_epoch=self._best_epoch,
            best_metric=self._best_metric or 0.0,
            best_checkpoint=self._best_checkpoint,
            history=self._history,
            gpu_id=self._gpu_id,
            hyperparams=self._cfg.hyperparams if self._cfg else None,
        )
