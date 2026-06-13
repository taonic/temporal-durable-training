"""Worker: registers all workflows + activities on the training task queue."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from temporalio.worker import Worker

from durable_training.activities.broker import GpuBrokerActivities
from durable_training.activities.training import (
    TrainingActivities,
    evaluate,
    prepare_data,
    register_model,
    simulate_step,
)
from durable_training.common import connect
from durable_training.shared import TASK_QUEUE
from durable_training.workflows.gpu_pool import GpuPoolWorkflow
from durable_training.workflows.sweep import SweepWorkflow
from durable_training.workflows.training import TrainingWorkflow


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    client = await connect()
    broker = GpuBrokerActivities()
    # train_epoch pings the GPU pool's health API, so give it the client.
    training = TrainingActivities(pool_client=client)

    # Thread pool runs the sync (CPU/GPU-bound) training activities; the async
    # broker activities run on the event loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        worker = Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[TrainingWorkflow, GpuPoolWorkflow, SweepWorkflow],
            activities=[
                simulate_step,
                prepare_data,
                training.train_epoch,
                evaluate,
                register_model,
                broker.acquire_gpu,
                broker.release_gpu,
            ],
            activity_executor=executor,
        )
        logging.info("Worker started on task queue %r. Ctrl-C to stop.", TASK_QUEUE)
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
