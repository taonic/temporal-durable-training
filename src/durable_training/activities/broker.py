"""GPU-pool broker activities.

Workflows can't synchronously call Updates/Queries on *other* workflows, so the
shared GpuPoolWorkflow is driven from activities that use a Temporal client.
This is the idiomatic way to do cross-workflow synchronous coordination.

``acquire_gpu`` blocks (via a long-running Update) until a slot is free; the
update id is derived from the run_id so a retried activity never double-leases.

NB: no ``from __future__ import annotations`` — Temporal decodes activity inputs
from real type hints; stringized hints would arrive as plain dicts.
"""

import asyncio

from temporalio import activity
from temporalio.client import Client, WorkflowUpdateStage

from durable_training.shared import (
    AcquireGpuInput,
    GpuLeaseRequest,
    ReleaseGpuInput,
)
from durable_training.workflows.gpu_pool import GpuPoolWorkflow


class GpuBrokerActivities:
    def __init__(
        self,
        target_host: str = "localhost:7233",
        namespace: str = "default",
        client: Client | None = None,
    ):
        self._target_host = target_host
        self._namespace = namespace
        self._client = client  # pre-built client (e.g. test env); else lazily connect

    async def _get_client(self) -> Client:
        if self._client is None:
            from temporalio.contrib.pydantic import pydantic_data_converter

            self._client = await Client.connect(
                self._target_host,
                namespace=self._namespace,
                data_converter=pydantic_data_converter,
            )
        return self._client

    @activity.defn
    async def acquire_gpu(self, input: AcquireGpuInput) -> int:
        from durable_training.common import ensure_gpu_pool

        client = await self._get_client()
        # Self-heal: (re)start the pool if it isn't running — e.g. after a dev-server
        # restart it would otherwise be "workflow not found for ID: gpu-pool".
        await ensure_gpu_pool(client, pool_id=input.pool_id)
        handle = client.get_workflow_handle(input.pool_id)
        # start_update(ACCEPTED) + result() is the robust pattern for a blocking
        # update (the handler waits until a GPU frees). The update id is derived
        # from the run_id so a retried activity reuses the same lease, never doubles.
        update = await handle.start_update(
            GpuPoolWorkflow.acquire,
            GpuLeaseRequest(run_id=input.run_id),
            id=f"acquire-{input.run_id}-{input.lease_seq}",
            wait_for_stage=WorkflowUpdateStage.ACCEPTED,
        )
        # The update blocks until a GPU frees. Heartbeat while we wait so the
        # activity stays alive under a short heartbeat timeout (and a worker crash
        # mid-wait reschedules to a fresh attempt that resumes the same update).
        result = asyncio.ensure_future(update.result())
        try:
            while True:
                done, _ = await asyncio.wait({result}, timeout=2.0)
                if result in done:
                    gpu_id = result.result()
                    activity.logger.info("Leased GPU %s to %s", gpu_id, input.run_id)
                    return gpu_id
                activity.heartbeat()
        except asyncio.CancelledError:
            result.cancel()
            raise

    @activity.defn
    async def release_gpu(self, input: ReleaseGpuInput) -> None:
        client = await self._get_client()
        handle = client.get_workflow_handle(input.pool_id)
        await handle.signal(GpuPoolWorkflow.release, input.gpu_id)
        activity.logger.info("Released GPU %s", input.gpu_id)
