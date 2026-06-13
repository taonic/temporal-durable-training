import uuid

from temporalio.client import WorkflowUpdateStage
from temporalio.worker import Worker

from durable_training.shared import GpuLeaseRequest, GpuPoolState
from durable_training.workflows.gpu_pool import GpuPoolWorkflow


async def test_lease_release_and_fifo_blocking(env, task_queue):
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[GpuPoolWorkflow],
    ):
        handle = await env.client.start_workflow(
            GpuPoolWorkflow.run,
            GpuPoolState(num_gpus=2, holders=[None, None]),
            id="pool-" + uuid.uuid4().hex[:8],
            task_queue=task_queue,
        )

        # Warm up so the first acquire update can't race the unstarted workflow.
        await handle.query(GpuPoolWorkflow.utilization)

        g0 = await handle.execute_update(
            GpuPoolWorkflow.acquire, GpuLeaseRequest(run_id="a")
        )
        g1 = await handle.execute_update(
            GpuPoolWorkflow.acquire, GpuLeaseRequest(run_id="b")
        )
        assert {g0, g1} == {0, 1}

        # Third acquire must block — both GPUs are leased.
        pending = await handle.start_update(
            GpuPoolWorkflow.acquire,
            GpuLeaseRequest(run_id="c"),
            wait_for_stage=WorkflowUpdateStage.ACCEPTED,
        )
        util = await handle.query(GpuPoolWorkflow.utilization)
        assert util.busy == 2 and util.queue_depth == 1

        # Free one GPU; the queued waiter gets exactly that slot.
        await handle.signal(GpuPoolWorkflow.release, g0)
        g2 = await pending.result()
        assert g2 == g0

        await handle.terminate()


async def test_resize_grows_pool_and_unblocks_waiter(env, task_queue):
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[GpuPoolWorkflow],
    ):
        handle = await env.client.start_workflow(
            GpuPoolWorkflow.run,
            GpuPoolState(num_gpus=1, holders=[None]),
            id="pool-" + uuid.uuid4().hex[:8],
            task_queue=task_queue,
        )
        await handle.query(GpuPoolWorkflow.utilization)

        g0 = await handle.execute_update(
            GpuPoolWorkflow.acquire, GpuLeaseRequest(run_id="a")
        )
        assert g0 == 0

        # Second acquire blocks — the single GPU is taken.
        pending = await handle.start_update(
            GpuPoolWorkflow.acquire,
            GpuLeaseRequest(run_id="b"),
            wait_for_stage=WorkflowUpdateStage.ACCEPTED,
        )
        assert (await handle.query(GpuPoolWorkflow.utilization)).queue_depth == 1

        # Grow the pool live — the queued waiter immediately gets the new GPU.
        await handle.signal(GpuPoolWorkflow.resize, 2)
        g1 = await pending.result()
        assert g1 == 1
        util = await handle.query(GpuPoolWorkflow.utilization)
        assert util.total == 2 and util.busy == 2 and util.queue_depth == 0

        await handle.terminate()
