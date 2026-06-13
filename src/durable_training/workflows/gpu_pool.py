"""GpuPoolWorkflow — a durable semaphore over a fixed set of GPUs.

A long-lived workflow that leases scarce GPUs to many training runs. FIFO queue
of waiters makes the contention fair and the queue depth observable, which is the
visual centerpiece of the GPU-utilization story: 2 GPUs, 6 jobs, stay saturated.

NB: no ``from __future__ import annotations`` — Temporal decodes update/run args
from real type hints (stringized hints would arrive as dicts).
"""

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from durable_training.shared import (
        GpuLeaseRequest,
        GpuPoolState,
        GpuSlot,
        GpuUtilization,
    )


@workflow.defn
class GpuPoolWorkflow:
    def __init__(self) -> None:
        self._num_gpus = 0
        self._holders: list[str | None] = []
        self._dead: set[int] = set()  # gpu_ids killed (failed hardware) — never leased
        self._queue: list[tuple[int, str]] = []  # (ticket, run_id) in FIFO order
        self._ticket_seq = 0

    @workflow.run
    async def run(self, state: GpuPoolState) -> None:
        self._num_gpus = state.num_gpus
        self._holders = list(state.holders)
        self._dead = set(state.dead)
        self._queue = [(i, rid) for i, rid in enumerate(state.queue)]
        self._ticket_seq = len(self._queue)

        # Serve forever; continue-as-new periodically to bound event history.
        await workflow.wait_condition(
            lambda: workflow.info().is_continue_as_new_suggested()
        )
        # Let any in-flight acquire handlers settle before rolling over.
        await workflow.wait_condition(workflow.all_handlers_finished)
        workflow.continue_as_new(
            GpuPoolState(
                num_gpus=self._num_gpus,
                holders=self._holders,
                queue=[rid for _, rid in self._queue],
                dead=sorted(self._dead),
            )
        )

    def _free_slot(self) -> int:
        for i, holder in enumerate(self._holders):
            if holder is None and i not in self._dead:
                return i
        return -1

    @workflow.update
    async def acquire(self, req: GpuLeaseRequest) -> int:
        """Block until a GPU is free and it's this waiter's turn, then lease it."""
        ticket = self._ticket_seq
        self._ticket_seq += 1
        self._queue.append((ticket, req.run_id))

        # NB: guard the index — wait_condition predicates are re-evaluated after
        # other waiters pop the queue, so it can be empty when this runs.
        await workflow.wait_condition(
            lambda: bool(self._queue)
            and self._queue[0][0] == ticket
            and self._free_slot() != -1
        )
        gpu = self._free_slot()
        self._holders[gpu] = req.run_id
        self._queue.pop(0)
        workflow.logger.info("GPU %d leased to %s", gpu, req.run_id)
        return gpu

    @workflow.signal
    def release(self, gpu_id: int) -> None:
        if 0 <= gpu_id < len(self._holders):
            workflow.logger.info("GPU %d released by %s", gpu_id, self._holders[gpu_id])
            self._holders[gpu_id] = None

    def _live_ids(self) -> list[int]:
        return [i for i in range(len(self._holders)) if i not in self._dead]

    @workflow.signal
    def kill_gpu(self, gpu_id: int) -> None:
        """Simulate a GPU failing: retire the slot (never reused) and free it. The
        job holding it finds out by *pinging* the GPU (see ``gpu_alive``) — there's
        no push signal to the training workflow."""
        if not (0 <= gpu_id < len(self._holders)) or gpu_id in self._dead:
            return
        holder = self._holders[gpu_id]
        self._dead.add(gpu_id)
        self._holders[gpu_id] = None
        self._num_gpus = len(self._live_ids())
        workflow.logger.info("GPU %d killed (held by %s)", gpu_id, holder)

    @workflow.query
    def gpu_alive(self, gpu_id: int) -> bool:
        """Health check a job pings while training. False once the GPU is killed."""
        return 0 <= gpu_id < len(self._holders) and gpu_id not in self._dead

    @workflow.signal
    def resize(self, num_gpus: int) -> None:
        """Grow or shrink the *live* pool. Growing appends free GPUs and wakes any
        queued waiters; shrinking only removes idle trailing GPUs (never a busy one)."""
        target = max(1, num_gpus)
        live = len(self._live_ids())
        while live < target:
            self._holders.append(None)
            live += 1
        while live > target and self._holders[-1] is None:
            idx = len(self._holders) - 1
            self._holders.pop()
            if idx in self._dead:
                self._dead.discard(idx)
            else:
                live -= 1
        self._num_gpus = len(self._live_ids())
        workflow.logger.info("GPU pool resized to %d", self._num_gpus)

    @workflow.query
    def utilization(self) -> GpuUtilization:
        live = self._live_ids()
        busy = sum(1 for i in live if self._holders[i] is not None)
        return GpuUtilization(
            total=len(live),
            busy=busy,
            free=len(live) - busy,
            queue_depth=len(self._queue),
            slots=[GpuSlot(gpu_id=i, holder=self._holders[i]) for i in live],
            waiting=[rid for _, rid in self._queue],
        )
