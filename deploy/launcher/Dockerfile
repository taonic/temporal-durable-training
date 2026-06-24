# The fly.io launcher image — a tiny FastAPI app that provisions Daytona
# sandboxes. Build context is the repo root (see fly.toml build.dockerfile).
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app

COPY deploy/launcher/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/launcher/main.py deploy/launcher/provision.py ./

EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
