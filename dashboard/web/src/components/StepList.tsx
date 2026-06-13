import type { JobStep } from "../types";

function Marker({ status }: { status: string }) {
  if (status === "done")
    return (
      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-300 text-xs">
        ✓
      </span>
    );
  if (status === "running")
    return (
      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-sky-500/20">
        <span className="h-3 w-3 rounded-full border-2 border-sky-300/30 border-t-sky-300 animate-spin" />
      </span>
    );
  return (
    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-800 ring-1 ring-white/10">
      <span className="h-1.5 w-1.5 rounded-full bg-slate-600" />
    </span>
  );
}

export function StepList({ steps }: { steps: JobStep[] }) {
  return (
    <ol className="mt-3 flex items-start">
      {steps.map((s, i) => {
        const active = s.status === "running";
        const done = s.status === "done";
        const first = i === 0;
        const last = i === steps.length - 1;
        return (
          <li key={s.name} className="flex-1 min-w-0 flex flex-col items-center">
            {/* connector + marker row */}
            <div className="flex items-center w-full">
              <span
                className={`h-0.5 flex-1 ${
                  first ? "opacity-0" : done || active ? "bg-emerald-400/50" : "bg-slate-700"
                }`}
              />
              <span className="mx-1 shrink-0">
                <Marker status={s.status} />
              </span>
              <span
                className={`h-0.5 flex-1 ${
                  last ? "opacity-0" : done ? "bg-emerald-400/50" : "bg-slate-700"
                }`}
              />
            </div>
            <span
              title={`${s.label} (${s.kind})`}
              className={`mt-1.5 px-1 text-center text-[10px] leading-tight line-clamp-2 ${
                active
                  ? "text-sky-300 font-medium"
                  : done
                  ? "text-slate-300"
                  : "text-slate-500"
              }`}
            >
              {s.label}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
