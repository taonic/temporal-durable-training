import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { GpuUtilization } from "../types";
import { resizeGpu, killGpu } from "../api";

export function GpuGrid({
  gpu,
  onInspect,
}: {
  gpu: GpuUtilization | null;
  onInspect?: (runId: string) => void;
}) {
  // GPU ids the user just killed — animate as failing until the slot is retired.
  const [killing, setKilling] = useState<Set<number>>(new Set());
  useEffect(() => {
    const ids = new Set((gpu?.slots ?? []).map((s) => s.gpu_id));
    setKilling((prev) => {
      const next = new Set([...prev].filter((id) => ids.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [gpu]);
  function onKill(id: number) {
    setKilling((prev) => new Set(prev).add(id));
    killGpu(id);
  }
  // Always render the pool — fall back to a default 2-idle-GPU view before the
  // pool's first utilization query arrives.
  const g: GpuUtilization = gpu ?? {
    total: 2,
    busy: 0,
    free: 2,
    queue_depth: 0,
    slots: [
      { gpu_id: 0, holder: null },
      { gpu_id: 1, holder: null },
    ],
    waiting: [],
  };
  return (
    <section className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">GPU Pool</h2>
        <div className="flex items-center gap-3">
          <div className="text-sm text-slate-400">
            <span className="text-emerald-400 font-semibold">{g.busy}</span>/{g.total} busy
            {g.queue_depth > 0 && (
              <span className="ml-3 rounded-full bg-amber-500/20 text-amber-300 px-2 py-0.5">
                {g.queue_depth} queued
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => resizeGpu(g.total - 1)}
              disabled={g.total <= 1}
              title="Remove an idle GPU"
              className="h-6 w-6 rounded-md ring-1 ring-white/15 text-slate-300 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer leading-none"
            >
              −
            </button>
            <button
              onClick={() => resizeGpu(g.total + 1)}
              title="Add a GPU (live, no restart)"
              className="h-6 w-6 rounded-md ring-1 ring-emerald-400/40 text-emerald-300 hover:bg-emerald-500/15 cursor-pointer leading-none"
            >
              +
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <AnimatePresence>
        {g.slots.map((s) => {
          const busy = !!s.holder;
          const dying = killing.has(s.gpu_id);
          return (
            <motion.div
              key={s.gpu_id}
              layout
              onClick={busy && !dying ? () => onInspect?.(s.holder!) : undefined}
              title={dying ? "GPU failing…" : busy ? `Inspect job ${s.holder}` : "idle"}
              animate={
                dying
                  ? { x: [0, -4, 4, -3, 3, 0] }
                  : {
                      boxShadow: busy
                        ? "0 0 24px rgba(16,185,129,0.45)"
                        : "0 0 0 rgba(0,0,0,0)",
                    }
              }
              exit={{ opacity: 0, scale: 0.85 }}
              className={`relative rounded-xl p-4 ring-1 transition-colors ${
                dying
                  ? "bg-rose-500/15 ring-rose-400/50"
                  : busy
                  ? "bg-emerald-500/10 ring-emerald-400/40 cursor-pointer hover:ring-emerald-300 hover:bg-emerald-500/20"
                  : "bg-slate-800/40 ring-white/10 cursor-default"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm text-slate-300">GPU {s.gpu_id}</span>
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    dying
                      ? "bg-rose-400"
                      : busy
                      ? "bg-emerald-400 animate-pulse"
                      : "bg-slate-600"
                  }`}
                />
              </div>
              <div className="mt-3 text-xs font-mono truncate text-slate-400">
                {dying ? "failing…" : s.holder ?? "idle"}
              </div>
              {busy && !dying && (
                <div className="mt-1 text-[10px] text-emerald-400/70">click to inspect →</div>
              )}
              {!dying && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onKill(s.gpu_id);
                  }}
                  title="Kill this GPU (simulate hardware failure)"
                  className="absolute top-1.5 right-1.5 h-4 w-4 rounded text-rose-300/70 hover:text-rose-200 hover:bg-rose-500/20 cursor-pointer text-xs leading-none"
                >
                  ✕
                </button>
              )}
            </motion.div>
          );
        })}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {g.waiting.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 flex flex-wrap gap-2"
          >
            <span className="text-xs text-slate-500 self-center">waiting:</span>
            {g.waiting.map((w) => (
              <span
                key={w}
                className="text-xs font-mono rounded-md bg-amber-500/10 text-amber-300 px-2 py-1"
              >
                {w}
              </span>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
