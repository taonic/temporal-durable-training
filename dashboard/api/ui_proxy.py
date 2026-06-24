"""Header-stripping reverse proxy for the Temporal Web UI.

The Temporal UI (http://localhost:8233) sends ``X-Frame-Options`` / a
``Content-Security-Policy`` (header *and* a ``<meta>`` tag) that stop it being
embedded in an iframe. This tiny proxy mirrors the UI at the root of its own port
(so the SPA's absolute asset paths still resolve) and drops those headers, letting
the dashboard slide it in as a panel. Same idea as the reference course sandbox.

Run:
  uv run python -m dashboard.api.ui_proxy          # serves on :8234 -> :8233
"""

from __future__ import annotations

import os
import re

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

TARGET = os.environ.get("TEMPORAL_UI_TARGET", "http://localhost:8233")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8234"))
# 0.0.0.0 in a sandbox so the Daytona preview can reach it; localhost otherwise.
PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")

# Headers that block framing or that httpx already decoded for us.
_DROP = {
    "x-frame-options",
    "content-security-policy",
    "content-security-policy-report-only",
    "content-encoding",
    "content-length",
    "transfer-encoding",
}
_CSP_META = re.compile(
    r'<meta[^>]*http-equiv=["\']?content-security-policy["\']?[^>]*>', re.IGNORECASE
)

app = FastAPI(title="Temporal UI proxy")
_client = httpx.AsyncClient(base_url=TARGET, follow_redirects=True, timeout=30.0)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(path: str, request: Request) -> Response:
    try:
        upstream = await _client.request(
            request.method,
            "/" + path,
            params=request.query_params,
            headers={
                k: v
                for k, v in request.headers.items()
                if k.lower()
                not in {"host", "accept-encoding", "connection", "content-length"}
            },
            content=await request.body(),
        )
    except httpx.ConnectError:
        return Response(
            content=(
                "<body style='font:14px sans-serif;background:#0a0e17;color:#e6edf3;"
                "padding:40px'>Temporal Web UI isn't reachable at "
                f"{TARGET}.<br>Start it with <code>temporal server start-dev</code>."
                "</body>"
            ),
            status_code=502,
            media_type="text/html",
        )

    body = upstream.content
    ctype = upstream.headers.get("content-type", "")
    # Also strip the CSP <meta> from HTML — its strict-dynamic would block scripts.
    if "text/html" in ctype:
        body = _CSP_META.sub("", body.decode("utf-8", "replace")).encode("utf-8")

    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _DROP}
    return Response(content=body, status_code=upstream.status_code, headers=headers)


def main() -> None:
    print(f"Temporal UI proxy: {PROXY_HOST}:{PROXY_PORT}  ->  {TARGET}")
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="warning")


if __name__ == "__main__":
    main()
