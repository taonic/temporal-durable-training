"""Start a pool of worker processes on the training task queue.

Each worker is a separate OS process (distinct Temporal identity), so they form a
real pool — which is what the node-failure demo needs: kill one mid-training and
the others pick up the rescheduled work.

  uv run python scripts/run_workers.py        # 2 workers (default)
  uv run python scripts/run_workers.py 4      # 4 workers

Ctrl-C stops them all. To demo a node dying, `kill -9 <pid>` one of the printed
PIDs while a training run is in flight (or use the injected `vanish_on_epoch` chaos).
"""

from __future__ import annotations

import subprocess
import sys
import time


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    procs: list[subprocess.Popen] = []
    for i in range(n):
        p = subprocess.Popen([sys.executable, "-m", "durable_training.worker"])
        procs.append(p)
        print(f"  worker {i}  pid {p.pid}")

    print(
        f"\n{n} workers polling 'training-task-queue'. Ctrl-C to stop all.\n"
        "Demo a node failure: `kill -9 <pid>` one worker mid-training — its work "
        "reschedules onto the others and resumes from the last checkpoint.\n"
    )

    reported: set[int] = set()
    try:
        while True:
            for p in procs:
                if p.poll() is not None and p.pid not in reported:
                    reported.add(p.pid)
                    print(f"  worker pid {p.pid} exited (code {p.returncode})")
            if all(p.poll() is not None for p in procs):
                print("all workers exited")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping workers...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
