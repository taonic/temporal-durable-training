import { useEffect, useState } from "react";

// Docked side panel: it slides in by growing its width (content clipped via
// overflow), so the dashboard's main column shrinks to make room rather than the
// UI floating over it. A drag handle on its left edge resizes it.
//
// `proxyUrl` is the header-stripping UI proxy (dashboard/api/ui_proxy.py): locally
// http://localhost:8234; in a Daytona sandbox it's the signed preview URL.
export function TemporalDrawer({
  path,
  proxyUrl,
  onClose,
}: {
  path: string | null;
  proxyUrl: string;
  onClose: () => void;
}) {
  const open = path !== null;
  const [width, setWidth] = useState(() =>
    Math.round(Math.min(window.innerWidth * 0.5, 900))
  );
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: PointerEvent) => {
      const w = window.innerWidth - e.clientX;
      setWidth(Math.max(360, Math.min(window.innerWidth - 280, w)));
    };
    const onUp = () => setDragging(false);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    document.body.style.userSelect = "none";
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.style.userSelect = "";
    };
  }, [dragging]);

  return (
    <aside
      className="shrink-0 overflow-hidden bg-slate-950 border-l border-white/10"
      style={{
        width: open ? width : 0,
        transition: dragging ? "none" : "width 300ms ease-out",
      }}
    >
      {open && (
        <div className="sticky top-0 h-screen flex" style={{ width }}>
          {/* drag handle */}
          <div
            onPointerDown={() => setDragging(true)}
            title="Drag to resize"
            className={`w-1.5 shrink-0 cursor-col-resize hover:bg-indigo-400/50 ${
              dragging ? "bg-indigo-400/60" : "bg-white/10"
            }`}
          />
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/10 shrink-0">
              <div className="flex items-center gap-2 min-w-0">
                <span className="h-2 w-2 rounded-full bg-indigo-400" />
                <span className="text-sm font-semibold">Temporal UI</span>
                <span className="text-xs text-slate-500 font-mono truncate">{path}</span>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={`${proxyUrl}${path}`}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-slate-400 hover:text-white cursor-pointer rounded-md ring-1 ring-white/15 px-2 py-1"
                >
                  new tab ↗
                </a>
                <button
                  onClick={onClose}
                  className="text-sm text-slate-400 hover:text-white cursor-pointer rounded-md ring-1 ring-white/15 px-2 py-1"
                >
                  close ✕
                </button>
              </div>
            </div>
            <iframe
              key={path}
              title="Temporal UI"
              src={`${proxyUrl}${path}`}
              className="flex-1 w-full bg-white"
              // don't let the iframe swallow pointer events mid-drag
              style={{ pointerEvents: dragging ? "none" : "auto" }}
            />
          </div>
        </div>
      )}
    </aside>
  );
}
