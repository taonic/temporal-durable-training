import { useState } from "react";
import { useDashboard, startSweep, startRun } from "./api";
import { GpuGrid } from "./components/GpuGrid";
import { WorkerPool } from "./components/WorkerPool";
import { TrainingPanel } from "./components/TrainingPanel";
import { SweepPanel } from "./components/SweepPanel";
import { TemporalDrawer } from "./components/TemporalDrawer";

export default function App() {
  const { state, connected } = useDashboard();
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [uiPath, setUiPath] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const openUi = (workflowId?: string) =>
    setUiPath(workflowId ? `/namespaces/default/workflows/${workflowId}` : "/");

  const selectedRun = selected
    ? state.runs.find((r) => r.run_id === selected) ?? null
    : null;

  const PAGE_SIZE = 6;

  const rand = () => Math.random().toString(36).slice(2, 8);
  const fire = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };

  const SCENARIOS: {
    key: string;
    title: string;
    desc: string;
    accent: string;
    run: () => Promise<void>;
  }[] = [
    {
      key: "standard",
      title: "Standard run",
      desc: "A clean training job through the full 7-step pipeline.",
      accent: "ring-emerald-400/40 hover:bg-emerald-500/10",
      run: () => startRun(`job-${rand()}`),
    },
    {
      key: "saturate",
      title: "Saturate GPUs",
      desc: "Launch 6 jobs at once so the GPU pool queues the overflow.",
      accent: "ring-sky-400/40 hover:bg-sky-500/10",
      run: async () => {
        for (let i = 0; i < 6; i++) await startRun(`gpu-${rand()}`);
      },
    },
    {
      key: "sweep",
      title: "Hyperparameter sweep",
      desc: "Fan out a grid of runs; the best model parks for human approval.",
      accent: "ring-violet-400/40 hover:bg-violet-500/10",
      run: () => startSweep(`sweep-${rand()}`),
    },
  ];

  // Standalone runs (not part of a sweep) for the GPU / fault-tolerance panels.
  const sweepRunIds = new Set(
    state.sweeps.flatMap((s) => s.leaderboard.map((e) => e.run_id))
  );
  const standaloneRuns = state.runs.filter((r) => !sweepRunIds.has(r.run_id));

  return (
    <div className="flex min-h-screen">
      <main className="flex-1 min-w-0 px-6 py-6">
      <div className="max-w-7xl mx-auto">
      <header className="flex items-center justify-between flex-wrap gap-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Durable Training
            <span className="ml-2 text-slate-500 text-base font-normal">
              on Temporal
            </span>
          </h1>
          <p className="text-sm text-slate-500">
            Fault-tolerant · GPU-efficient · human-in-the-loop ML training
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`text-xs flex items-center gap-1.5 ${
              connected ? "text-emerald-400" : "text-rose-400"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                connected ? "bg-emerald-400" : "bg-rose-400"
              }`}
            />
            {connected ? "live" : "reconnecting"}
          </span>
          <button
            onClick={() => openUi()}
            className="text-sm rounded-lg ring-1 ring-white/15 px-3 py-1.5 hover:bg-white/5 cursor-pointer"
          >
            Temporal UI ⇥
          </button>
        </div>
      </header>

      <div className="mb-6">
        <button
          disabled={busy}
          onClick={() => setShowCreate(true)}
          className="rounded-lg bg-indigo-500/20 text-indigo-200 px-4 py-2 text-sm font-medium hover:bg-indigo-500/30 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
        >
          + Create job
        </button>
      </div>

      {/* Top: GPU pool + Worker pool, side by side */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <GpuGrid gpu={state.gpu} onInspect={setSelected} />
        <WorkerPool workers={state.workers} />
      </div>

      {/* Sweeps (when present) */}
      {state.sweeps.length > 0 && (
        <div className="space-y-6 mb-6">
          {state.sweeps.map((s) => (
            <SweepPanel key={s.name} sweep={s} />
          ))}
        </div>
      )}

      {/* Training runs, full width underneath */}
      <div>
          {(() => {
            const pageCount = Math.max(1, Math.ceil(standaloneRuns.length / PAGE_SIZE));
            const safePage = Math.min(page, pageCount - 1);
            const pageRuns = standaloneRuns.slice(
              safePage * PAGE_SIZE,
              safePage * PAGE_SIZE + PAGE_SIZE
            );
            return (
              <>
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold">
                    Training runs
                    {standaloneRuns.length > 0 && (
                      <span className="ml-2 text-sm font-normal text-slate-500">
                        {standaloneRuns.length}
                      </span>
                    )}
                  </h2>
                  {pageCount > 1 && (
                    <div className="flex items-center gap-2 text-sm">
                      <button
                        onClick={() => setPage((p) => Math.max(0, p - 1))}
                        disabled={safePage === 0}
                        className="rounded-md ring-1 ring-white/15 px-2 py-1 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                      >
                        ‹ prev
                      </button>
                      <span className="text-slate-400">
                        {safePage + 1} / {pageCount}
                      </span>
                      <button
                        onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                        disabled={safePage >= pageCount - 1}
                        className="rounded-md ring-1 ring-white/15 px-2 py-1 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                      >
                        next ›
                      </button>
                    </div>
                  )}
                </div>
                {standaloneRuns.length === 0 ? (
                  <div className="rounded-2xl bg-slate-900/40 ring-1 ring-white/10 p-8 text-center text-slate-500">
                    No active runs. Use the buttons above, or run a demo script.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-4">
                    {pageRuns.map((r) => (
                      <TrainingPanel
                        key={r.run_id}
                        run={r}
                        onOpenUi={() => openUi(r.run_id)}
                      />
                    ))}
                  </div>
                )}
              </>
            );
          })()}
      </div>

      {selected && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center bg-black/80 p-6 overflow-y-auto"
          onClick={() => setSelected(null)}
        >
          <div
            className="w-full max-w-xl mt-12 rounded-2xl bg-slate-950 ring-1 ring-white/10 shadow-2xl p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm text-slate-400">
                Job on GPU · <span className="font-mono text-slate-200">{selected}</span>
              </h3>
              <button
                onClick={() => setSelected(null)}
                className="text-slate-400 hover:text-white cursor-pointer text-sm rounded-md ring-1 ring-white/15 px-2 py-1"
              >
                close ✕
              </button>
            </div>
            {selectedRun ? (
              <TrainingPanel run={selectedRun} onOpenUi={() => openUi(selected!)} />
            ) : (
              <div className="rounded-2xl bg-slate-900/60 ring-1 ring-white/10 p-6 text-sm text-slate-400">
                This job is no longer reporting progress (it may have finished and
                released the GPU). Open it in the Temporal UI for full history.
                <a
                  href={`${state.temporal_ui}/namespaces/default/workflows/${selected}`}
                  target="_blank"
                  rel="noreferrer"
                  className="block mt-2 text-sky-300 hover:underline cursor-pointer"
                >
                  open in Temporal UI ↗
                </a>
              </div>
            )}
          </div>
        </div>
      )}
      </div>
      </main>

      {showCreate && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center bg-black/80 p-6 overflow-y-auto"
          onClick={() => setShowCreate(false)}
        >
          <div
            className="w-full max-w-lg mt-16 rounded-2xl bg-slate-950 ring-1 ring-white/10 shadow-2xl p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-semibold">Create a job</h3>
              <button
                onClick={() => setShowCreate(false)}
                className="text-sm text-slate-400 hover:text-white cursor-pointer rounded-md ring-1 ring-white/15 px-2 py-1"
              >
                close ✕
              </button>
            </div>
            <p className="text-sm text-slate-500 mb-4">Pick a scenario to run.</p>
            <div className="space-y-2.5">
              {SCENARIOS.map((sc) => (
                <button
                  key={sc.key}
                  disabled={busy}
                  onClick={() => {
                    setShowCreate(false);
                    fire(sc.run);
                  }}
                  className={`w-full text-left rounded-xl bg-slate-900/60 ring-1 ${sc.accent} p-3 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer transition-colors`}
                >
                  <div className="text-sm font-medium text-slate-100">{sc.title}</div>
                  <div className="mt-0.5 text-xs text-slate-400">{sc.desc}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <TemporalDrawer path={uiPath} onClose={() => setUiPath(null)} />
    </div>
  );
}
