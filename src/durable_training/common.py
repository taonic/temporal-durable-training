"""Client helpers shared by the worker, demo scripts, and dashboard API."""

from __future__ import annotations

from temporalio.client import Client, WorkflowHandle
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.service import RPCError

from durable_training.shared import GPU_POOL_WORKFLOW_ID, TASK_QUEUE, GpuPoolState
from durable_training.workflows.gpu_pool import GpuPoolWorkflow

DEFAULT_TARGET = "localhost:7233"
DEFAULT_NAMESPACE = "default"


async def connect(
    target: str = DEFAULT_TARGET, namespace: str = DEFAULT_NAMESPACE
) -> Client:
    """Connect with the Pydantic data converter so our models serialize cleanly."""
    return await Client.connect(
        target, namespace=namespace, data_converter=pydantic_data_converter
    )


async def ensure_gpu_pool(
    client: Client,
    num_gpus: int = 2,
    pool_id: str = GPU_POOL_WORKFLOW_ID,
    reset: bool = False,
) -> WorkflowHandle:
    """Start the GPU pool if absent (get-or-create). ``reset`` restarts it fresh."""
    if reset:
        try:
            await client.get_workflow_handle(pool_id).terminate("resetting gpu pool")
        except RPCError:
            pass
    handle = await client.start_workflow(
        GpuPoolWorkflow.run,
        GpuPoolState(num_gpus=num_gpus, holders=[None] * num_gpus),
        id=pool_id,
        task_queue=TASK_QUEUE,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
    )
    # Warm up: a query forces the first workflow task to be processed, so the
    # very first acquire() update can't race an unstarted workflow.
    await handle.query(GpuPoolWorkflow.utilization)
    return handle
