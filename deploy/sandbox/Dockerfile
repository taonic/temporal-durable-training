# The image a Daytona sandbox runs: the full Durable Training stack — Temporal
# dev server, worker(s), the dashboard API, and the header-stripping UI proxy.
# The launcher (on fly.io) creates one of these per visitor and execs boot.sh.

# --- Stage 1: build the SPA bundle (served by the dashboard API) -------------
FROM node:20-bookworm-slim AS web
WORKDIR /web
COPY dashboard/web/package.json dashboard/web/package-lock.json* ./
RUN npm install
COPY dashboard/web/ ./
RUN npm run build   # -> /web/dist

# --- Stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

# curl (health checks) + the Temporal CLI (dev server).
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates bash \
 && rm -rf /var/lib/apt/lists/* \
 && curl -sSf https://temporal.download/cli.sh | sh \
 && cp /root/.temporalio/bin/temporal /usr/local/bin/temporal \
 && pip install --no-cache-dir uv

WORKDIR /app
# Install Python deps first (cached), then the project.
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
RUN uv sync --extra dashboard

# The built SPA, served by the dashboard API's StaticFiles mount.
COPY --from=web /web/dist ./dashboard/web/dist
COPY deploy/sandbox/boot.sh /app/boot.sh
RUN chmod +x /app/boot.sh

# Ports: 8000 dashboard API (SPA + /api + /ws), 8234 Temporal UI proxy,
# 8233 Temporal UI (internal), 7233 Temporal frontend (internal).
EXPOSE 8000 8234

# Daytona keeps the container alive and execs boot.sh into it; if run directly,
# boot the stack.
CMD ["/app/boot.sh"]
