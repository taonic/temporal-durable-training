import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { WorkerInfo } from "../types";
import { addWorker, killWorker } from "../api";

export function WorkerPool({ workers }: { workers: WorkerInfo[] }) {
  // Hide workers the OS reports as dead (alive === false), even though Temporal's
  // poller list still lists them for ~a minute. `null` = remote host (can't ps).
  const sorted = [...workers]
    .filter((w) => w.alive !== false)
    .sort((a, b) => a.identity.localeCompare(b.identity));
  // PIDs the user just killed — animate them as "terminating" until the poller
  // entry ages out of the task queue and the card leaves for good.
  const [killing, setKilling] = useState<Set<number>>(new Set());

  // Drop killed PIDs from local state once they're gone from the worker list.
  useEffect(() => {
    setKilling((prev) => {
      const present = new Set(workers.map((w) => w.pid));
      const next = new Set([...prev].filter((pid) => present.has(pid)));
      return next.size === prev.size ? prev : next;
    });
  }, [workers]);

  function onKill(pid: number) {
    setKilling((prev) => new Set(prev).add(pid));
    killWorker(pid);
  }

  return (
    <section className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">
          Worker Pool
          <span className="ml-2 text-sm font-normal text-slate-500">{sorted.length}</span>
        </h2>
        <button
          onClick={() => addWorker()}
          title="Spawn a new worker process on this host"
          className="text-sm rounded-md ring-1 ring-emerald-400/40 text-emerald-300 px-2 py-1 hover:bg-emerald-500/15 cursor-pointer"
        >
          + Add worker
        </button>
      </div>

      {sorted.length === 0 ? (
        <div className="text-sm text-slate-500">
          No workers polling. Add one, or run{" "}
          <span className="font-mono">scripts/run_workers.py</span>.
        </div>
      ) : (
        <div className="space-y-2">
          <AnimatePresence>
            {sorted.map((w) => {
              const dying = w.pid != null && killing.has(w.pid);
              return (
                <motion.div
                  key={w.identity}
                  layout
                  initial={{ opacity: 0, y: 4 }}
                  animate={
                    dying
                      ? { opacity: 0.7, x: [0, -5, 5, -4, 4, 0] }
                      : { opacity: 1, x: 0, y: 0 }
                  }
                  exit={{ opacity: 0, scale: 0.9, height: 0, marginBottom: 0 }}
                  transition={dying ? { duration: 0.4 } : undefined}
                  className={`flex items-center justify-between rounded-lg px-3 py-2 ring-1 ${
                    dying
                      ? "bg-rose-500/15 ring-rose-400/50"
                      : "bg-slate-800/40 ring-white/10"
                  }`}
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span
                        className={`h-2 w-2 rounded-full ${
                          dying ? "bg-rose-400" : "bg-emerald-400 animate-pulse"
                        }`}
                      />
                      <span
                        className={`font-mono text-sm truncate ${
                          dying ? "line-through text-rose-200/80" : ""
                        }`}
                      >
                        {w.identity}
                      </span>
                    </div>
                    <div className="mt-1 flex gap-1.5">
                      {dying ? (
                        <span className="text-[10px] rounded bg-rose-500/20 text-rose-300 px-1.5">
                          terminating…
                        </span>
                      ) : (
                        <>
                          {w.workflow && (
                            <span className="text-[10px] rounded bg-sky-500/15 text-sky-300 px-1.5">
                              workflow
                            </span>
                          )}
                          {w.activity && (
                            <span className="text-[10px] rounded bg-violet-500/15 text-violet-300 px-1.5">
                              activity
                            </span>
                          )}
                          {w.spawned && (
                            <span className="text-[10px] rounded bg-emerald-500/15 text-emerald-300 px-1.5">
                              dashboard-spawned
                            </span>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                  {w.spawned && w.pid != null && !dying && (
                    <button
                      onClick={() => onKill(w.pid!)}
                      title="Kill this worker (simulate a node going away)"
                      className="shrink-0 text-xs rounded-md ring-1 ring-rose-400/40 text-rose-300 px-2 py-1 hover:bg-rose-500/15 cursor-pointer"
                    >
                      kill
                    </button>
                  )}
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}
      <p className="mt-3 text-xs text-slate-600">
        Kill a worker mid-training and its work reschedules onto the others — resuming
        from the last checkpoint.
      </p>
    </section>
  );
}
