"""SweepWorkflow — a reusable hyperparameter sweep with a human-approval gate.

Fans out one child TrainingWorkflow per hyperparameter combination; the shared
GPU pool naturally bounds how many run at once. When all finish, the sweep picks
the best model and *parks for a human*: it waits — durably, surviving worker
restarts, with a review timeout — for an approve/reject decision before promoting
the model. A DAG can't pause for a human and resume in place; this can.

NB: no ``from __future__ import annotations`` — Temporal decodes the SweepConfig
argument from the run method's real type hints (stringized hints arrive as dicts).
"""

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from durable_training.activities.training import register_model
    from durable_training.shared import (
        ApprovalDecision,
        Hyperparams,
        LeaderboardEntry,
        PendingApproval,
        RegisteredModel,
        RegisterModelInput,
        SweepConfig,
        SweepResult,
        SweepStatus,
        TrainingConfig,
        TrainingInput,
        TrainingResult,
    )
    from durable_training.workflows.training import TrainingWorkflow


@workflow.defn
class SweepWorkflow:
    def __init__(self) -> None:
        self._cfg: SweepConfig | None = None
        self._status = "running"
        self._configs: list[TrainingConfig] = []
        self._results: dict[str, TrainingResult] = {}
        self._candidate: PendingApproval | None = None
        self._decision: ApprovalDecision | None = None
        self._registered: RegisteredModel | None = None

    @workflow.run
    async def run(self, sweep: SweepConfig) -> SweepResult:
        self._cfg = sweep
        self._configs = self._expand(sweep)
        self._status = "running"

        async def run_child(index: int, cfg: TrainingConfig) -> TrainingResult:
            # No explicit task_queue: children inherit the parent's, so they run
            # on whatever worker is hosting the sweep (correct in prod and tests).
            res = await workflow.execute_child_workflow(
                TrainingWorkflow.run,
                TrainingInput(config=cfg),
                id=cfg.run_id,
            )
            self._results[cfg.run_id] = res
            return res

        results = await asyncio.gather(
            *[run_child(i, c) for i, c in enumerate(self._configs)]
        )

        # Select the best model by validation accuracy.
        best = max(results, key=lambda r: r.best_metric)
        self._candidate = PendingApproval(
            candidate_run_id=best.run_id,
            model_name=sweep.base.model_name,
            metric=best.best_metric,
            hyperparams=best.hyperparams or Hyperparams(),
            checkpoint_path=best.best_checkpoint,
        )

        # --- Human-in-the-loop approval gate ---
        self._status = "pending_approval"
        if sweep.auto_decision is not None:
            self._decision = ApprovalDecision(
                approved=sweep.auto_decision == "approve",
                reviewer="auto",
                note="auto_decision",
            )
        else:
            try:
                await workflow.wait_condition(
                    lambda: self._decision is not None,
                    timeout=timedelta(seconds=sweep.review_timeout_seconds),
                )
            except asyncio.TimeoutError:
                self._decision = ApprovalDecision(
                    approved=False, reviewer="auto", note="review timeout"
                )

        assert self._decision is not None
        if self._decision.approved and best.best_checkpoint:
            self._registered = await workflow.execute_activity(
                register_model,
                RegisterModelInput(
                    run_id=best.run_id,
                    model_name=sweep.base.model_name,
                    checkpoint_path=best.best_checkpoint,
                    metric=best.best_metric,
                    reviewer=self._decision.reviewer,
                    note=self._decision.note,
                ),
                start_to_close_timeout=timedelta(minutes=5),
            )
            self._status = "approved"
        else:
            self._status = "rejected"

        return SweepResult(
            name=sweep.name,
            status=self._status,
            best=best,
            registered=self._registered,
            decision=self._decision,
            results=results,
        )

    @staticmethod
    def _expand(sweep: SweepConfig) -> list[TrainingConfig]:
        configs: list[TrainingConfig] = []
        i = 0
        for lr in sweep.search_space.learning_rate:
            for bs in sweep.search_space.batch_size:
                cfg = sweep.base.model_copy(deep=True)
                cfg.run_id = f"{sweep.name}-{i}"
                cfg.hyperparams = Hyperparams(learning_rate=lr, batch_size=bs)
                cfg.register_on_complete = False
                configs.append(cfg)
                i += 1
        return configs

    # --- signals / queries --------------------------------------------------

    @workflow.signal
    def approve(self, reviewer: str, note: str = "") -> None:
        self._decision = ApprovalDecision(approved=True, reviewer=reviewer, note=note)

    @workflow.signal
    def reject(self, reviewer: str, note: str = "") -> None:
        self._decision = ApprovalDecision(approved=False, reviewer=reviewer, note=note)

    @workflow.query
    def status(self) -> SweepStatus:
        leaderboard: list[LeaderboardEntry] = []
        for cfg in self._configs:
            res = self._results.get(cfg.run_id)
            leaderboard.append(
                LeaderboardEntry(
                    run_id=cfg.run_id,
                    hyperparams=cfg.hyperparams,
                    best_metric=res.best_metric if res else 0.0,
                    status=res.status if res else "running",
                )
            )
        leaderboard.sort(key=lambda e: e.best_metric, reverse=True)
        return SweepStatus(
            name=self._cfg.name if self._cfg else "",
            status=self._status,
            total=len(self._configs),
            completed=len(self._results),
            leaderboard=leaderboard,
            pending_approval=self._candidate if self._status == "pending_approval" else None,
            decision=self._decision,
        )
