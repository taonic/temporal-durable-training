import { motion, AnimatePresence } from "framer-motion";
import type { RunProgress } from "../types";
import { signalRun } from "../api";
import { StepList } from "./StepList";

const STATUS_COLORS: Record<string, string> = {
  training: "text-sky-300 bg-sky-500/15",
  completed: "text-emerald-300 bg-emerald-500/15",
  needs_attention: "text-amber-300 bg-amber-500/15",
  acquiring_gpu: "text-violet-300 bg-violet-500/15",
  preparing: "text-slate-300 bg-slate-500/15",
};

export function TrainingPanel({
  run,
  onOpenUi,
}: {
  run: RunProgress;
  onOpenUi?: () => void;
}) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-mono text-sm truncate">{run.run_id}</h3>
          <span
            className={`mt-1 inline-block text-xs rounded-full px-2 py-0.5 ${
              STATUS_COLORS[run.status] ?? "text-slate-300 bg-slate-500/15"
            }`}
          >
            {run.status} · epoch {Math.min(run.current_epoch + 1, run.max_epochs)}/
            {run.max_epochs}
          </span>
        </div>
        <div className="flex items-start gap-3">
          {onOpenUi && (
            <button
              onClick={onOpenUi}
              className="text-xs text-slate-400 hover:text-sky-300 cursor-pointer rounded-md ring-1 ring-white/15 px-2 py-1 whitespace-nowrap"
            >
              Temporal UI ⇥
            </button>
          )}
          <div className="text-right">
            <div className="text-xs text-slate-500">best acc</div>
            <div className="text-lg font-semibold text-emerald-300">
              {run.best_metric != null ? run.best_metric.toFixed(3) : "—"}
            </div>
          </div>
        </div>
      </div>

      {/* Prominent: which GPU and which worker this run is on. */}
      <div className="mt-3 flex flex-wrap gap-2">
        <div className="flex items-center gap-2 rounded-lg bg-emerald-500/15 ring-1 ring-emerald-400/40 px-3 py-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400/80">
            GPU
          </span>
          <span className="text-base font-bold font-mono text-emerald-200 leading-none">
            {run.gpu_id != null && run.gpu_id >= 0 ? run.gpu_id : "—"}
          </span>
        </div>
        <div
          className="flex items-center gap-2 rounded-lg bg-indigo-500/15 ring-1 ring-indigo-400/40 px-3 py-1.5 min-w-0"
          title={run.worker ? `Executing on worker ${run.worker}` : "no worker assigned"}
        >
          <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-400/80">
            Worker
          </span>
          <span className="text-sm font-bold font-mono text-indigo-200 leading-none truncate">
            {run.worker ?? "—"}
          </span>
        </div>
      </div>

      <AnimatePresence>
        {(() => {
          const completed = run.history?.length ?? 0;
          // Mid-run GPU re-lease (revoked → acquiring) counts as recovery too.
          const reacquiring =
            (run.status === "gpu_revoked" || run.status === "acquiring_gpu") &&
            completed > 0;
          if (!run.retrying && !reacquiring) return null;
          return (
            <motion.div
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="mt-3 rounded-lg bg-rose-500/15 ring-1 ring-rose-400/40 px-3 py-2 text-sm text-rose-200"
            >
              ⚡ Recovering{run.retrying ? ` — activity retry #${run.retry_attempt}` : ""}.
              <span className="font-semibold">
                {" "}Resuming from epoch {run.current_epoch + 1}
              </span>{" "}
              via checkpoint — {completed} earlier epoch
              {completed === 1 ? "" : "s"} kept, not restarted from epoch 0.
              {run.last_failure ? (
                <span className="block text-xs text-rose-300/70 font-mono truncate">
                  {run.last_failure}
                </span>
              ) : null}
            </motion.div>
          );
        })()}
      </AnimatePresence>

      {run.steps && run.steps.length > 0 && <StepList steps={run.steps} />}

      <AnimatePresence>
        {run.needs_attention && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="mt-3 rounded-lg bg-amber-500/15 ring-1 ring-amber-400/40 p-3"
          >
            <div className="text-sm text-amber-200">
              Needs attention at epoch {run.needs_attention.epoch}:{" "}
              {run.needs_attention.reason}
            </div>
            <div className="mt-2 flex gap-2">
              <button
                onClick={() => signalRun(run.run_id, "resume_decision", ["continue", 0])}
                className="cursor-pointer text-xs rounded bg-sky-500/20 text-sky-200 px-2 py-1 hover:bg-sky-500/30"
              >
                Continue
              </button>
              <button
                onClick={() => signalRun(run.run_id, "resume_decision", ["adjust_lr", 0.01])}
                className="cursor-pointer text-xs rounded bg-violet-500/20 text-violet-200 px-2 py-1 hover:bg-violet-500/30"
              >
                Set LR 0.01
              </button>
              <button
                onClick={() => signalRun(run.run_id, "resume_decision", ["stop", 0])}
                className="cursor-pointer text-xs rounded bg-rose-500/20 text-rose-200 px-2 py-1 hover:bg-rose-500/30"
              >
                Stop
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
