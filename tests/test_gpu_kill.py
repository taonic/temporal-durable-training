import asyncio
import uuid

from temporalio.worker import Worker

from durable_training.activities.broker import GpuBrokerActivities
from durable_training.shared import (
    GpuPoolState,
    TrainingConfig,
    TrainingInput,
)
from durable_training.workflows.gpu_pool import GpuPoolWorkflow
from durable_training.workflows.training import TrainingWorkflow

from .conftest import training_activities


async def test_kill_gpu_revokes_and_job_reacquires(env, task_queue, executor):
    """Killing the GPU a job holds revokes its lease; the job drops its in-flight
    epoch, leases a healthy GPU, and resumes from the last checkpoint."""
    broker = GpuBrokerActivities(client=env.client)
    pool_id = "pool-" + uuid.uuid4().hex[:8]

    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[GpuPoolWorkflow, TrainingWorkflow],
        activities=[
            *training_activities(env.client),
            broker.acquire_gpu,
            broker.release_gpu,
        ],
        activity_executor=executor,
    ):
        pool = await env.client.start_workflow(
            GpuPoolWorkflow.run,
            GpuPoolState(num_gpus=2, holders=[None, None]),
            id=pool_id,
            task_queue=task_queue,
        )
        await pool.query(GpuPoolWorkflow.utilization)  # warm up

        run_id = "g-" + uuid.uuid4().hex[:8]
        cfg = TrainingConfig(
            run_id=run_id,
            require_gpu=True,
            gpu_pool_id=pool_id,
            pipeline_seconds=0.0,  # instant preamble; we want to catch the training step
            step_seconds=0.2,  # slow enough to catch it mid-training
            steps_per_epoch=6,
            max_epochs=4,
        )
        handle = await env.client.start_workflow(
            TrainingWorkflow.run,
            TrainingInput(config=cfg),
            id=run_id,
            task_queue=task_queue,
        )

        # Wait until the job has leased a GPU and is training, then kill that GPU.
        gpu_id = None
        for _ in range(100):
            prog = await handle.query(TrainingWorkflow.progress)
            if prog.status == "training" and prog.gpu_id is not None:
                gpu_id = prog.gpu_id
                break
            await asyncio.sleep(0.1)
        assert gpu_id is not None
        await pool.signal(GpuPoolWorkflow.kill_gpu, gpu_id)

        result = await handle.result()
        assert result.status == "completed"
        assert [m.epoch for m in result.history] == [0, 1, 2, 3]
        # It re-acquired the surviving GPU (the killed one is retired).
        assert result.gpu_id != gpu_id

        util = await pool.query(GpuPoolWorkflow.utilization)
        assert util.total == 1  # one GPU was retired
        await pool.terminate()
