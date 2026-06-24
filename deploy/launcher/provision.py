"""Daytona sandbox provisioning, shared by the fly.io launcher (main.py) and the
local dev CLI (local_launch.py).

The flow is identical wherever it runs: create a sandbox from SANDBOX_IMAGE, mint
signed preview URLs for :8000 (dashboard) and :8234 (UI proxy), exec boot.sh with
the browser-reachable UI-proxy URL injected, and wait until the dashboard answers.
Only the trigger differs — fly serves a landing page; the CLI just calls it.

Env:
  DAYTONA_KEY      Daytona API key (required)
  SANDBOX_IMAGE    image the sandbox runs (built from deploy/sandbox/Dockerfile)
  DAYTONA_TARGET   optional Daytona target/region

NB: the Daytona Python SDK surface shifts between versions — pin `daytona` in
requirements.txt and adjust the create/preview/exec calls to match if needed.
"""

from __future__ import annotations

import os
import time

import httpx
from daytona import (
    CreateSandboxFromImageParams,
    Daytona,
    DaytonaConfig,
    Resources,
    SessionExecuteRequest,
)

SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "durable-training-sandbox:latest")
DASHBOARD_PORT = 8000
UI_PROXY_PORT = 8234
BOOT_TIMEOUT_S = 150


def daytona_client() -> Daytona:
    """Build a Daytona client from the environment. Raises if DAYTONA_KEY is unset."""
    key = os.environ.get("DAYTONA_KEY", "")
    if not key:
        raise RuntimeError("DAYTONA_KEY not set")
    cfg: dict = {"api_key": key}
    target = os.environ.get("DAYTONA_TARGET")
    if target:
        cfg["target"] = target
    return Daytona(DaytonaConfig(**cfg))


def provision(daytona: Daytona) -> dict:
    """Create a sandbox, boot the full stack, and wait until the dashboard is up.

    Returns ``{"sandbox": <handle>, "sandbox_id": str, "url": str}`` where ``url``
    is the dashboard's signed preview URL. On any failure the half-booted sandbox
    is deleted before the error propagates, so callers never leak one.
    """
    sandbox = daytona.create(
        CreateSandboxFromImageParams(
            image=SANDBOX_IMAGE,
            resources=Resources(cpu=2, memory=2, disk=5),
            # Reclaim idle/old per-visitor sandboxes automatically (minutes).
            auto_stop_interval=15,
            auto_delete_interval=60,
        )
    )

    try:
        dashboard = sandbox.get_preview_link(DASHBOARD_PORT)
        ui_proxy = sandbox.get_preview_link(UI_PROXY_PORT)

        # Boot the stack with the *browser-reachable* UI-proxy URL injected so the
        # dashboard can hand the Temporal-UI iframe a URL that works from outside.
        sandbox.process.create_session("stack")
        sandbox.process.execute_session_command(
            "stack",
            SessionExecuteRequest(
                command=(
                    f"export TEMPORAL_UI_PROXY_URL='{ui_proxy.url}'; "
                    f"bash /app/boot.sh"
                ),
                run_async=True,
            ),
        )

        wait_until_ready(dashboard.url, getattr(dashboard, "token", None))
        return {"sandbox": sandbox, "sandbox_id": sandbox.id, "url": dashboard.url}
    except Exception:
        # Don't leave a half-booted sandbox lying around.
        try:
            sandbox.delete()
        except Exception:
            pass
        raise


def wait_until_ready(url: str, token: str | None) -> None:
    """Poll ``{url}/api/health`` until it answers (<500) or BOOT_TIMEOUT_S elapses."""
    headers = {"x-daytona-preview-token": token} if token else {}
    deadline = time.time() + BOOT_TIMEOUT_S
    last = ""
    with httpx.Client(timeout=8.0, follow_redirects=True) as client:
        while time.time() < deadline:
            try:
                r = client.get(f"{url}/api/health", headers=headers)
                if r.status_code < 500:
                    return
                last = f"status {r.status_code}"
            except Exception as e:  # not up yet
                last = str(e)
            time.sleep(2)
    raise RuntimeError(f"dashboard never became ready ({last})")
