import { motion, AnimatePresence } from "framer-motion";
import type { SweepStatus } from "../types";
import { decideSweep } from "../api";

export function SweepPanel({ sweep }: { sweep: SweepStatus }) {
  const best = sweep.leaderboard[0];
  return (
    <motion.div
      layout
      className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-5"
    >
      <div className="flex items-center justify-between">
        <h3 className="font-mono text-sm">{sweep.name}</h3>
        <span className="text-xs text-slate-400">
          {sweep.completed}/{sweep.total} runs · {sweep.status}
        </span>
      </div>

      <table className="mt-3 w-full text-sm">
        <thead>
          <tr className="text-slate-500 text-xs text-left">
            <th className="font-normal py-1">run</th>
            <th className="font-normal">lr</th>
            <th className="font-normal">bs</th>
            <th className="font-normal text-right">val acc</th>
            <th className="font-normal text-right">status</th>
          </tr>
        </thead>
        <tbody>
          {sweep.leaderboard.map((e) => (
            <tr
              key={e.run_id}
              className={e === best && e.best_metric > 0 ? "text-emerald-300" : "text-slate-300"}
            >
              <td className="font-mono py-1 truncate max-w-[120px]">
                {e === best && e.best_metric > 0 ? "★ " : ""}
                {e.run_id}
              </td>
              <td className="font-mono">{e.hyperparams.learning_rate}</td>
              <td className="font-mono">{e.hyperparams.batch_size}</td>
              <td className="font-mono text-right">
                {e.best_metric > 0 ? e.best_metric.toFixed(3) : "—"}
              </td>
              <td className="text-right text-xs text-slate-400">{e.status}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <AnimatePresence>
        {sweep.pending_approval && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-4 rounded-xl bg-amber-500/10 ring-1 ring-amber-400/40 p-4"
          >
            <div className="text-amber-200 font-semibold text-sm">
              ⏸ Human approval required
            </div>
            <div className="mt-1 text-sm text-slate-300">
              Promote{" "}
              <span className="font-mono">{sweep.pending_approval.candidate_run_id}</span>{" "}
              (val acc {sweep.pending_approval.metric.toFixed(3)}, lr{" "}
              {sweep.pending_approval.hyperparams.learning_rate}) to production?
            </div>
            <div className="mt-3 flex gap-2">
              <button
                onClick={() => decideSweep(sweep.name, true)}
                className="cursor-pointer rounded-lg bg-emerald-500/20 text-emerald-200 px-3 py-1.5 text-sm hover:bg-emerald-500/30"
              >
                Approve
              </button>
              <button
                onClick={() => decideSweep(sweep.name, false)}
                className="cursor-pointer rounded-lg bg-rose-500/20 text-rose-200 px-3 py-1.5 text-sm hover:bg-rose-500/30"
              >
                Reject
              </button>
            </div>
            <div className="mt-2 text-xs text-slate-500">
              The workflow is waiting durably — it survives worker restarts until you decide.
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {sweep.decision && (
        <div
          className={`mt-3 text-xs ${
            sweep.decision.approved ? "text-emerald-400" : "text-rose-400"
          }`}
        >
          {sweep.decision.approved ? "✓ approved" : "✗ rejected"} by{" "}
          {sweep.decision.reviewer}
          {sweep.decision.note ? ` — "${sweep.decision.note}"` : ""}
        </div>
      )}
    </motion.div>
  );
}
