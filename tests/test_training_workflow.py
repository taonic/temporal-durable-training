import asyncio
import uuid

from temporalio.worker import Worker

from durable_training.shared import ChaosConfig, TrainingConfig, TrainingInput
from durable_training.workflows.training import TrainingWorkflow

from .conftest import TRAINING_ACTIVITIES


def _cfg(**kw) -> TrainingConfig:
    base = dict(
        run_id="t-" + uuid.uuid4().hex[:8],
        require_gpu=False,
        step_seconds=0.0,
        pipeline_seconds=0.0,
        max_epochs=4,
        steps_per_epoch=3,
    )
    base.update(kw)
    return TrainingConfig(**base)


async def test_completes_and_improves(env, task_queue, executor):
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[TrainingWorkflow],
        activities=TRAINING_ACTIVITIES,
        activity_executor=executor,
    ):
        cfg = _cfg()
        res = await env.client.execute_workflow(
            TrainingWorkflow.run, TrainingInput(config=cfg), id=cfg.run_id, task_queue=task_queue
        )
        assert res.status == "completed"
        assert [m.epoch for m in res.history] == [0, 1, 2, 3]
        # Validation accuracy increases each epoch in the sim, so the last is best.
        assert res.best_epoch == 3

        # The full pipeline ran and every step completed.
        prog = await env.client.get_workflow_handle(cfg.run_id).query(
            TrainingWorkflow.progress
        )
        names = [s.name for s in prog.steps]
        assert names == ["interpret", "build", "quota", "choose", "reserve", "spec", "train"]
        assert all(s.status == "done" for s in prog.steps)


async def test_crash_resumes_from_checkpoint(env, task_queue, executor):
    """A crash at epoch 2 is retried; epochs 0-1 are not recomputed, run completes."""
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[TrainingWorkflow],
        activities=TRAINING_ACTIVITIES,
        activity_executor=executor,
    ):
        cfg = _cfg(chaos=ChaosConfig(crash_on_epoch=2, crash_attempts=1))
        res = await env.client.execute_workflow(
            TrainingWorkflow.run, TrainingInput(config=cfg), id=cfg.run_id, task_queue=task_queue
        )
        assert res.status == "completed"
        # Every epoch is present exactly once despite the injected crash.
        assert [m.epoch for m in res.history] == [0, 1, 2, 3]


async def test_human_in_the_loop_intervention(env, task_queue, executor):
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[TrainingWorkflow],
        activities=TRAINING_ACTIVITIES,
        activity_executor=executor,
    ):
        cfg = _cfg(pause_for_review_on_epoch=1)
        handle = await env.client.start_workflow(
            TrainingWorkflow.run, TrainingInput(config=cfg), id=cfg.run_id, task_queue=task_queue
        )

        # Wait until the run parks for a human at epoch 1.
        prog = None
        for _ in range(200):
            prog = await handle.query(TrainingWorkflow.progress)
            if prog.needs_attention is not None:
                break
            await asyncio.sleep(0.05)
        assert prog is not None and prog.needs_attention is not None
        assert prog.needs_attention.epoch == 1

        # Human says continue; the run finishes.
        await handle.signal(TrainingWorkflow.resume_decision, args=["continue", 0.0])
        res = await handle.result()
        assert res.status == "completed"
