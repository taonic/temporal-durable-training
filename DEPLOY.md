# Deploying to fly.io with Daytona sandboxes

The demo deploys as **two pieces**:

- A tiny **launcher** on **fly.io** (`deploy/launcher`) — serves a landing page and,
  per visitor, provisions a Daytona sandbox and redirects the browser to it.
- The **full stack** runs inside a per-visitor **Daytona sandbox**
  (`deploy/sandbox`) — Temporal dev server + worker(s) + dashboard API + the
  header-stripping UI proxy + the built SPA. Everything is same-origin on the
  sandbox's signed preview URL, so the SPA's `/api`, `/ws`, and the embedded
  Temporal UI all just work.

```
browser ──▶ fly launcher ──(Daytona SDK)──▶ create sandbox, boot stack
   │                                          ├─ temporal server start-dev (:7233/:8233)
   └──── redirect to ───────────────────────▶ ├─ dashboard API + SPA + /ws  (:8000)  ← preview URL
        sandbox :8000 preview URL             ├─ UI proxy (:8234)                    ← preview URL
                                              └─ worker(s)
```

Why this split: a Temporal **client** can't gRPC over an HTTP preview URL, so we
keep everything Temporal-touching inside the sandbox and only proxy HTTP/WS out.

## 1. Build & push the sandbox image

Daytona pulls the sandbox image from a registry. Build it from the repo root and
push to a registry your Daytona account can read (Docker Hub, GHCR, ECR, …):

```bash
docker build -f deploy/sandbox/Dockerfile -t <registry>/durable-training-sandbox:latest .
docker push <registry>/durable-training-sandbox:latest
```

This bakes the project (via `uv sync`), the Temporal CLI, and the built SPA, with
`deploy/sandbox/boot.sh` as the boot script.

## 2. Deploy the launcher to fly.io

```bash
fly launch --no-deploy            # or: fly apps create durable-training
fly secrets set DAYTONA_KEY=dtn_...                       # your Daytona API key
fly secrets set SANDBOX_IMAGE=<registry>/durable-training-sandbox:latest
fly deploy
```

`fly.toml` builds `deploy/launcher/Dockerfile` (FastAPI + the Daytona SDK) and
exposes it on :8080 with a `/api/health` check.

## 3. Use it

Open `https://durable-training.fly.dev`, click **Launch demo** → the launcher
creates a sandbox, boots the stack (~30–60s), and redirects you to the live
dashboard running in your sandbox. Sandboxes auto-stop after 15 min idle and
auto-delete after 60 min (`auto_stop_interval` / `auto_delete_interval` in
`deploy/launcher/main.py`).

## Run through Daytona from a local dev

You don't need to deploy the launcher to fly to exercise the real Daytona path.
`deploy/launcher/local_launch.py` runs the *exact* provisioning flow the launcher
uses (shared `deploy/launcher/provision.py`) straight from your laptop: it creates
a sandbox from `SANDBOX_IMAGE`, boots the stack, and prints the dashboard preview
URL. You still need the sandbox image in a registry Daytona can pull (step 1).

```bash
pip install -r deploy/launcher/requirements.txt   # daytona SDK + httpx (use a venv)
export DAYTONA_KEY=dtn_...
export SANDBOX_IMAGE=<registry>/durable-training-sandbox:latest
# export DAYTONA_TARGET=...                        # optional region/target

python deploy/launcher/local_launch.py             # create, boot, print URL; Ctrl-C deletes
python deploy/launcher/local_launch.py --open      # also open the dashboard in a browser
python deploy/launcher/local_launch.py --keep      # leave the sandbox running on exit
```

The URL is printed on stdout (logs go to stderr), so it's scriptable:
`URL=$(python deploy/launcher/local_launch.py --keep)`. Without `--keep` the
sandbox is deleted when you Ctrl-C; otherwise it auto-stops after 15 min idle and
auto-deletes after 60 min, same as the fly path.

To exercise the launcher's HTTP surface locally instead (landing page + `/api/session`),
run it as a server — same env vars: `cd deploy/launcher && uvicorn main:app --port 8080`.

## Local sanity check (no fly/Daytona)

The sandbox image runs the whole stack standalone, so you can verify it locally
without touching Daytona at all:

```bash
docker build -f deploy/sandbox/Dockerfile -t durable-training-sandbox .
docker run --rm -p 8000:8000 -p 8234:8234 durable-training-sandbox
# open http://localhost:8000
```

## Caveats / things to verify against your accounts

- **Daytona SDK version.** The SDK's surface (`create` / `get_preview_link` /
  `execute_session_command`) shifts across versions. Pin `daytona` in
  `deploy/launcher/requirements.txt` to the version matching `main.py`, and adjust
  the call shapes if your version differs.
- **Preview-URL auth.** If your Daytona preview links require a token, the dashboard
  and UI-proxy origins differ (two ports → two signed URLs); the launcher already
  forwards a `x-daytona-preview-token` header on its health probe. The browser must
  also send it — if your Daytona setup gates previews, front them or disable the gate
  for these ports.
- **WebSocket over preview.** The dashboard's `/ws` rides the sandbox's :8000 preview
  origin. Daytona dev-server previews support WS; if yours doesn't, the dashboard
  degrades to whatever the socket can deliver — switch `/ws` to polling if needed.
- **Cost.** Per-visitor sandboxes (cpu 2 / mem 2 / disk 5) add up; tighten the
  auto-stop/delete intervals or switch to a single shared sandbox for a driven demo.
