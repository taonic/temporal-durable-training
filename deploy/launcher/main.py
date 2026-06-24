"""fly.io launcher: provisions a per-visitor Daytona sandbox running the full
Durable Training stack and redirects the browser to it.

Flow:
  GET  /              -> landing page with a "Launch demo" button
  POST /api/session   -> create a Daytona sandbox from SANDBOX_IMAGE, mint signed
                         preview URLs for :8000 (dashboard) and :8234 (UI proxy),
                         exec boot.sh with the UI-proxy URL injected, wait until the
                         dashboard answers, and return its URL
  GET  /api/health    -> fly health check

The whole app (SPA + API + ws + Temporal + UI proxy) runs *inside* the sandbox, so
once redirected the visitor is entirely on the sandbox's preview origin. The
launcher stays tiny and stateless.

Env:
  DAYTONA_KEY      Daytona API key (required)
  SANDBOX_IMAGE    image the sandbox runs (built from deploy/sandbox/Dockerfile)
  DAYTONA_TARGET   optional Daytona target/region
  PORT             fly internal port (default 8080)

NB: the Daytona Python SDK surface shifts between versions — pin `daytona` in
requirements.txt and adjust the create/preview/exec calls to match if needed.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from provision import daytona_client, provision

app = FastAPI(title="Durable Training launcher")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/session")
def create_session() -> dict:
    """Create a sandbox, boot the stack, and return its dashboard URL."""
    if not os.environ.get("DAYTONA_KEY"):
        raise HTTPException(500, "DAYTONA_KEY not set")
    try:
        result = provision(daytona_client())
    except Exception as e:
        raise HTTPException(502, f"sandbox boot failed: {e}") from e
    return {"sandbox_id": result["sandbox_id"], "url": result["url"]}


_LANDING = """<!doctype html><html><head><meta charset=utf-8>
<title>Durable Training on Temporal</title>
<style>
  html,body{height:100%;margin:0;background:#0a0e17;color:#e6edf3;
    font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
  .wrap{height:100%;display:flex;flex-direction:column;align-items:center;
    justify-content:center;gap:20px;text-align:center;padding:24px}
  h1{font-size:28px;margin:0}
  p{color:#94a3b8;max-width:520px}
  button{font:600 15px sans-serif;color:#c7d2fe;background:rgba(99,102,241,.2);
    border:1px solid rgba(129,140,248,.5);border-radius:10px;padding:12px 22px;cursor:pointer}
  button:hover{background:rgba(99,102,241,.3)}
  button:disabled{opacity:.5;cursor:default}
  .spin{width:26px;height:26px;border-radius:50%;border:3px solid rgba(231,235,242,.25);
    border-top-color:#818cf8;animation:s .8s linear infinite}
  @keyframes s{to{transform:rotate(360deg)}}
  .err{color:#fca5a5}
</style></head><body><div class=wrap>
  <h1>Durable Training <span style="color:#64748b;font-weight:400">on Temporal</span></h1>
  <p>Launch a private sandbox running the full demo — Temporal, GPU pool,
     worker pool, and the live dashboard. Takes ~30–60s to boot.</p>
  <div id=ctl><button id=go onclick=launch()>Launch demo</button></div>
  <script>
    async function launch(){
      const ctl=document.getElementById('ctl');
      ctl.innerHTML='<div class=spin></div><p>Booting your sandbox…</p>';
      try{
        const r=await fetch('/api/session',{method:'POST'});
        if(!r.ok) throw new Error(await r.text());
        const {url}=await r.json();
        location.href=url;
      }catch(e){
        ctl.innerHTML='<p class=err>Launch failed: '+e.message+'</p>'+
          '<button onclick=launch()>Try again</button>';
      }
    }
  </script>
</div></body></html>"""


@app.get("/", response_class=HTMLResponse)
def landing() -> str:
    return _LANDING
