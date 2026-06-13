"""Shared Pydantic models and constants.

These models are the contracts that flow across workflow/activity boundaries.
They are intentionally dependency-light (pydantic only) so they can be imported
safely inside the workflow sandbox via ``workflow.unsafe.imports_passed_through``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# --- Topology constants -----------------------------------------------------

TASK_QUEUE = "training-task-queue"
GPU_POOL_WORKFLOW_ID = "gpu-pool"

Backend = Literal["sim", "torch"]


# --- Configuration models ---------------------------------------------------


class ChaosConfig(BaseModel):
    """Failure injection knobs that make fault tolerance demonstrable on demand."""

    crash_on_epoch: Optional[int] = None
    """Raise inside ``train_epoch`` when this epoch is reached (simulates a crash)."""
    crash_attempts: int = 1
    """How many attempts should crash before the epoch is allowed to succeed."""


class Hyperparams(BaseModel):
    learning_rate: float = 0.01
    batch_size: int = 32

    def slug(self) -> str:
        return f"lr{self.learning_rate}_bs{self.batch_size}"


class TrainingConfig(BaseModel):
    """Everything one training run needs."""

    run_id: str
    model_name: str = "demo-model"
    dataset: str = "synthetic"
    hyperparams: Hyperparams = Field(default_factory=Hyperparams)

    max_epochs: int = 5
    steps_per_epoch: int = 5
    early_stop_patience: int = 3

    backend: Backend = "sim"
    step_seconds: float = 1.0
    """Wall-clock per step in the sim backend, to mimic GPU compute time. With the
    default 5 steps this makes each epoch ~5s, while still heartbeating every step
    (~1s, under the 2s heartbeat timeout)."""

    heartbeat_timeout_seconds: int = 2
    """How long without a heartbeat before Temporal reschedules train_epoch onto
    another worker (the node-failure recovery knob). train_epoch heartbeats every
    step, so a tight 2s makes node failures detected/rescheduled quickly."""

    pipeline_seconds: float = 1.0
    """Scales the simulated delay of the pre-training pipeline steps (build image,
    query quota, reserve machines, …). Training itself is always the longest step."""

    require_gpu: bool = True
    gpu_pool_id: str = GPU_POOL_WORKFLOW_ID

    # Continue-as-new after this many epochs to keep event history bounded.
    continue_as_new_every: int = 1000

    register_on_complete: bool = False
    """Standalone runs register their own best model; sweeps gate it behind approval."""

    pause_for_review_on_epoch: Optional[int] = None
    """HITL divergence-intervention demo: pause for a human at this epoch."""

    chaos: ChaosConfig = Field(default_factory=ChaosConfig)


# --- Result / metric models -------------------------------------------------


class EpochMetrics(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    checkpoint_path: str


class TrainingResult(BaseModel):
    run_id: str
    status: str
    best_epoch: int
    best_metric: float
    best_checkpoint: Optional[str]
    history: list[EpochMetrics]
    gpu_id: Optional[int] = None
    hyperparams: Optional[Hyperparams] = None


class AttentionRequest(BaseModel):
    """Set when a run pauses for a human (divergence intervention)."""

    epoch: int
    reason: str
    val_loss: float


class JobStep(BaseModel):
    """One step of a training job's pipeline (rendered as the per-run progress)."""

    name: str
    label: str
    kind: str  # "workflow" | "activity" | "history"
    status: str = "pending"  # "pending" | "running" | "done"


class TrainingResumeState(BaseModel):
    """Carried across continue-as-new so a long run keeps its place and its GPU."""

    next_epoch: int
    gpu_id: Optional[int]
    lease_seq: int = 0
    steps: list["JobStep"] = Field(default_factory=list)
    best_metric: Optional[float]
    best_epoch: int
    best_checkpoint: Optional[str]
    latest_checkpoint: Optional[str]
    history: list["EpochMetrics"]
    patience: int


class TrainingInput(BaseModel):
    """Single argument to TrainingWorkflow.run.

    Temporal decodes workflow args from the run method's type hints; a single
    typed input object is the robust idiom (multi-arg / top-level Optional
    signatures can fail to decode and silently arrive as dicts).
    """

    config: TrainingConfig
    resume: Optional["TrainingResumeState"] = None


class TrainingProgress(BaseModel):
    """Query result powering the dashboard's live training panel."""

    run_id: str
    status: str
    current_epoch: int
    max_epochs: int
    best_metric: Optional[float]
    best_epoch: int
    latest_checkpoint: Optional[str]
    gpu_id: Optional[int]
    history: list[EpochMetrics]
    steps: list[JobStep] = Field(default_factory=list)
    needs_attention: Optional[AttentionRequest] = None


# --- Activity I/O models ----------------------------------------------------


class TrainEpochInput(BaseModel):
    config: TrainingConfig
    epoch: int
    resume_from: Optional[str]
    gpu_id: int
    lr_override: Optional[float] = None


class TrainEpochResult(BaseModel):
    epoch: int
    train_loss: float
    checkpoint_path: str


class EvaluateInput(BaseModel):
    config: TrainingConfig
    epoch: int
    checkpoint_path: str
    gpu_id: int


class EvaluateResult(BaseModel):
    epoch: int
    val_loss: float
    val_accuracy: float


class RegisterModelInput(BaseModel):
    run_id: str
    model_name: str
    checkpoint_path: str
    metric: float
    reviewer: str = "system"
    note: str = ""


class RegisteredModel(BaseModel):
    model_name: str
    version: str
    source_run_id: str
    metric: float
    path: str


# --- GPU pool models --------------------------------------------------------


class GpuLeaseRequest(BaseModel):
    run_id: str


class GpuSlot(BaseModel):
    gpu_id: int
    holder: Optional[str] = None  # run_id holding this GPU, or None if free


class GpuUtilization(BaseModel):
    total: int
    busy: int
    free: int
    queue_depth: int
    slots: list[GpuSlot]
    waiting: list[str]


class GpuPoolState(BaseModel):
    """Carried across continue-as-new so the pool never forgets its leases."""

    num_gpus: int
    holders: list[Optional[str]]  # index == gpu_id
    queue: list[str] = Field(default_factory=list)  # run_ids waiting, in FIFO order
    dead: list[int] = Field(default_factory=list)  # gpu_ids killed (failed) — never reused


# --- Broker activity I/O ----------------------------------------------------


class AcquireGpuInput(BaseModel):
    pool_id: str
    run_id: str
    lease_seq: int = 0
    """Bumped on each (re)acquire so a fresh lease isn't served the cached result
    of a prior acquire (which matters after a GPU is revoked)."""


class ReleaseGpuInput(BaseModel):
    pool_id: str
    gpu_id: int


# --- Sweep / HITL models ----------------------------------------------------


class SearchSpace(BaseModel):
    learning_rate: list[float] = Field(default_factory=lambda: [0.001, 0.01, 0.1])
    batch_size: list[int] = Field(default_factory=lambda: [32])


class SweepConfig(BaseModel):
    name: str
    base: TrainingConfig
    search_space: SearchSpace = Field(default_factory=SearchSpace)
    review_timeout_seconds: int = 600
    """How long the approval gate waits for a human before auto-rejecting."""
    auto_decision: Optional[Literal["approve", "reject"]] = None
    """For unattended demos/tests: resolve the gate without a human."""


class ApprovalDecision(BaseModel):
    approved: bool
    reviewer: str
    note: str = ""


class PendingApproval(BaseModel):
    candidate_run_id: str
    model_name: str
    metric: float
    hyperparams: Hyperparams
    checkpoint_path: Optional[str]


class LeaderboardEntry(BaseModel):
    run_id: str
    hyperparams: Hyperparams
    best_metric: float
    status: str


class SweepStatus(BaseModel):
    """Query result powering the dashboard's sweep leaderboard + approval card."""

    name: str
    status: str  # running | pending_approval | approved | rejected
    total: int
    completed: int
    leaderboard: list[LeaderboardEntry]
    pending_approval: Optional[PendingApproval] = None
    decision: Optional[ApprovalDecision] = None


class SweepResult(BaseModel):
    name: str
    status: str
    best: Optional[TrainingResult]
    registered: Optional[RegisteredModel]
    decision: Optional[ApprovalDecision]
    results: list[TrainingResult]
