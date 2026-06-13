import uuid

from temporalio.worker import Worker

from durable_training.shared import SearchSpace, SweepConfig, TrainingConfig
from durable_training.workflows.sweep import SweepWorkflow
from durable_training.workflows.training import TrainingWorkflow

from .conftest import TRAINING_ACTIVITIES


async def test_sweep_picks_best_and_approves(env, task_queue, executor):
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[SweepWorkflow, TrainingWorkflow],
        activities=TRAINING_ACTIVITIES,
        activity_executor=executor,
    ):
        base = TrainingConfig(
            run_id="ignored",
            model_name="test-sweep",
            require_gpu=False,
            step_seconds=0.0, pipeline_seconds=0.0,
            max_epochs=3,
            steps_per_epoch=2,
        )
        sweep = SweepConfig(
            name="sw-" + uuid.uuid4().hex[:8],
            base=base,
            search_space=SearchSpace(learning_rate=[0.001, 0.01], batch_size=[32]),
            auto_decision="approve",
        )
        res = await env.client.execute_workflow(
            SweepWorkflow.run, sweep, id=sweep.name, task_queue=task_queue
        )
        assert res.status == "approved"
        assert res.best is not None
        # lr=0.01 is the simulator's sweet spot, so it should win.
        assert res.best.hyperparams.learning_rate == 0.01
        assert res.registered is not None


async def test_sweep_rejection_skips_registration(env, task_queue, executor):
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[SweepWorkflow, TrainingWorkflow],
        activities=TRAINING_ACTIVITIES,
        activity_executor=executor,
    ):
        base = TrainingConfig(
            run_id="ignored",
            model_name="test-sweep-rej",
            require_gpu=False,
            step_seconds=0.0, pipeline_seconds=0.0,
            max_epochs=2,
            steps_per_epoch=2,
        )
        sweep = SweepConfig(
            name="swr-" + uuid.uuid4().hex[:8],
            base=base,
            search_space=SearchSpace(learning_rate=[0.01], batch_size=[32]),
            auto_decision="reject",
        )
        res = await env.client.execute_workflow(
            SweepWorkflow.run, sweep, id=sweep.name, task_queue=task_queue
        )
        assert res.status == "rejected"
        assert res.registered is None
