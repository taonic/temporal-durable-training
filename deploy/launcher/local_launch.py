#!/usr/bin/env python3
"""Provision a Daytona sandbox from your laptop — the exact flow the fly.io
launcher uses, minus fly. Use this to exercise the *real* Daytona path during
development without deploying the launcher.

This still requires the sandbox image to live in a registry Daytona can pull
(see DEPLOY.md step 1); it only replaces the fly-hosted trigger with a local one.

    pip install -r deploy/launcher/requirements.txt   # daytona SDK + httpx (in a venv)
    export DAYTONA_KEY=dtn_...
    export SANDBOX_IMAGE=<registry>/durable-training-sandbox:latest
    python deploy/launcher/local_launch.py            # create, boot, print URL; Ctrl-C deletes
    python deploy/launcher/local_launch.py --open      # also open the dashboard in a browser
    python deploy/launcher/local_launch.py --keep      # leave the sandbox running on exit
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser

# Run from anywhere: make `provision` importable the same way main.py imports it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from provision import SANDBOX_IMAGE, daytona_client, provision  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--keep", action="store_true", help="don't delete the sandbox on exit"
    )
    ap.add_argument(
        "--open", action="store_true", help="open the dashboard URL in a browser"
    )
    args = ap.parse_args()

    try:
        daytona = daytona_client()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"[local] creating sandbox from {SANDBOX_IMAGE} …", file=sys.stderr)
    result = provision(daytona)
    sandbox, url = result["sandbox"], result["url"]
    print(f"[local] sandbox {result['sandbox_id']} ready", file=sys.stderr)
    print(url)  # stdout: the dashboard preview URL, scriptable

    if args.open:
        webbrowser.open(url)

    if args.keep:
        print(
            f"[local] leaving sandbox {result['sandbox_id']} running (--keep); "
            "it auto-stops after 15 min idle, auto-deletes after 60 min",
            file=sys.stderr,
        )
        return 0

    try:
        print("[local] Ctrl-C to delete the sandbox and exit", file=sys.stderr)
        while True:
            input()  # block until EOF/Ctrl-C without busy-waiting
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print(f"[local] deleting sandbox {result['sandbox_id']} …", file=sys.stderr)
        try:
            sandbox.delete()
        except Exception as e:
            print(f"[local] delete failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
