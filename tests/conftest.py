import concurrent.futures
import uuid

import pytest
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

from durable_training.activities.training import (
    TrainingActivities,
    evaluate,
    prepare_data,
    register_model,
    simulate_step,
)


def training_activities(pool_client=None):
    """Activity list with a train_epoch bound to ``pool_client`` (needed only for
    GPU runs, where train_epoch pings the pool's health API)."""
    return [
        simulate_step,
        prepare_data,
        TrainingActivities(pool_client=pool_client).train_epoch,
        evaluate,
        register_model,
    ]


# Default (no pool client) — fine for require_gpu=False tests, which never ping.
TRAINING_ACTIVITIES = training_activities()


@pytest.fixture(autouse=True)
def _tmp_cwd(tmp_path, monkeypatch):
    """Keep checkpoints/models out of the repo during tests."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
async def env():
    # start_local (real dev server, wall-clock time) rather than time-skipping:
    # our activities do real work, and the time-skipping clock would race past
    # their heartbeat/start-to-close timeouts. None of these tests rely on
    # skipping durable timers (the approval gate uses auto_decision; pauses wait
    # on signals), so a local server is both correct and simpler.
    e = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
    try:
        yield e
    finally:
        await e.shutdown()


@pytest.fixture
def task_queue():
    return str(uuid.uuid4())


@pytest.fixture
def executor():
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        yield ex
