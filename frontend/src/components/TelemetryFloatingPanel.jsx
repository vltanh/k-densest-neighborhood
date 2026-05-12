import { useCallback, useEffect, useRef, useState } from 'react';
import { X, GripVertical } from 'lucide-react';
import { DispatchView, LogView, StatusBadge } from './TelemetryPanel';

const PANEL_W = 360;
const PANEL_H = 480;
const MIN_W = 280;
const MIN_H = 320;
const EDGE_INSET = 16;   // gap from parent right edge for default placement
const TOP_OFFSET = 64;   // gap below GraphView header strip

function Tab({ label, active, onClick, badge = null }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`relative pb-2 pt-1 eyebrow flex items-center gap-2 transition-colors duration-200 ease-out ${
        active ? 'text-[var(--on-night)]' : 'text-[var(--on-night-faint)] hover:text-[var(--on-night-dim)]'
      }`}
    >
      <span>{label}</span>
      {badge}
      <span
        className="absolute left-0 right-0 -bottom-px h-[2px] transition-[background-color,transform] duration-200 ease-out origin-left"
        style={{
          backgroundColor: active ? 'var(--gold)' : 'transparent',
          transform: active ? 'scaleX(1)' : 'scaleX(0.4)',
        }}
      />
    </button>
  );
}

// Floating, draggable telemetry overlay. Lives inside the GraphView container
// (a `relative` parent) and is absolutely positioned within it. The drag
// handle is the header strip; the body and tabs never receive drag events
// so the underlying GraphView stays interactive even when the panel is open.
export default function TelemetryFloatingPanel({
  telemetry, logs, loading, open = true, onClose,
}) {
  const [tab, setTab] = useState('dispatch');
  const [pos, setPos] = useState(null); // {x, y} once measured
  const dragRef = useRef(null);          // { startX, startY, originX, originY }
  const panelRef = useRef(null);

  // Default placement: top-right of the parent container, dropped below the
  // GraphView header strip. The ref callback fires synchronously on mount,
  // so we can read the parent's width before paint and seed `pos` from there.
  const setPanelRef = useCallback((el) => {
    panelRef.current = el;
    if (!el) return;
    setPos(prev => {
      if (prev) return prev;
      const parent = el.parentElement;
      const parentW = parent?.clientWidth || window.innerWidth;
      const pw = el.offsetWidth || PANEL_W;
      const x = Math.max(0, parentW - pw - EDGE_INSET);
      return { x, y: TOP_OFFSET };
    });
  }, []);

  const onHeaderMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    if (e.target.closest('[data-no-drag="true"]')) return;
    if (!pos) return;
    e.preventDefault();
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      originX: pos.x,
      originY: pos.y,
    };

    const onMove = (ev) => {
      const d = dragRef.current;
      if (!d) return;
      const dx = ev.clientX - d.startX;
      const dy = ev.clientY - d.startY;
      const parent = panelRef.current?.parentElement;
      const panelEl = panelRef.current;
      const pw = panelEl?.offsetWidth ?? PANEL_W;
      const ph = panelEl?.offsetHeight ?? PANEL_H;
      const maxX = Math.max(0, (parent?.clientWidth ?? window.innerWidth) - pw);
      const maxY = Math.max(0, (parent?.clientHeight ?? window.innerHeight) - ph);
      const nextX = Math.max(0, Math.min(maxX, d.originX + dx));
      const nextY = Math.max(0, Math.min(maxY, d.originY + dy));
      setPos({ x: nextX, y: nextY });
    };
    const onUp = () => {
      dragRef.current = null;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [pos]);

  // If the parent resizes (sidebar collapse, viewport shrink) and the panel
  // ends up partially off-screen, nudge it back inside.
  useEffect(() => {
    const panel = panelRef.current;
    const parent = panel?.parentElement;
    if (!panel || !parent) return;
    const ro = new ResizeObserver(() => {
      const pw = panel.offsetWidth;
      const ph = panel.offsetHeight;
      const maxX = Math.max(0, parent.clientWidth - pw);
      const maxY = Math.max(0, parent.clientHeight - ph);
      setPos(p => p ? ({
        x: Math.max(0, Math.min(maxX, p.x)),
        y: Math.max(0, Math.min(maxY, p.y)),
      }) : p);
    });
    ro.observe(parent);
    return () => ro.disconnect();
  }, []);

  return (
    <div
      ref={setPanelRef}
      className="absolute z-20 texture-night text-[var(--on-night)] border border-[var(--rule-night-2)] shadow-[6px_6px_0_0_rgba(5,11,20,0.45)] flex flex-col fade-in"
      style={{
        left: pos?.x ?? 0,
        top: pos?.y ?? TOP_OFFSET,
        width: PANEL_W,
        height: PANEL_H,
        minWidth: MIN_W,
        minHeight: MIN_H,
        display: open ? 'flex' : 'none',
        visibility: pos ? 'visible' : 'hidden',
      }}
      role="dialog"
      aria-label="Telemetry panel"
    >
      {/* Header — drag handle, status, close */}
      <div
        onMouseDown={onHeaderMouseDown}
        className="shrink-0 flex items-center justify-between gap-3 px-4 py-2 border-b border-[var(--rule-night)] bg-[var(--night-2)] cursor-grab active:cursor-grabbing select-none"
        title="Drag to move"
      >
        <div className="flex items-center gap-2 min-w-0">
          <GripVertical size={13} className="text-[var(--on-night-faint)] shrink-0" />
          <span className="eyebrow text-[var(--on-night-dim)] truncate">Telemetry</span>
        </div>
        <div className="flex items-center gap-3 shrink-0" data-no-drag="true">
          <StatusBadge status={telemetry?.status} />
          <button
            type="button"
            onClick={onClose}
            className="text-[var(--on-night-faint)] hover:text-[var(--on-night)] transition-colors p-1 -mr-1"
            title="Close telemetry"
            aria-label="Close telemetry panel"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Tab bar */}
      <div className="shrink-0 px-4 pt-3">
        <div className="flex items-center gap-7 border-b border-[var(--rule-night)]">
          <Tab label="Dispatch" active={tab === 'dispatch'} onClick={() => setTab('dispatch')} />
          <Tab
            label="Log"
            active={tab === 'log'}
            onClick={() => setTab('log')}
            badge={
              logs.length > 0 && (
                <span className="text-[length:var(--text-xs)] font-mono tnum text-[var(--on-night-faint)] normal-case tracking-normal">
                  {logs.length}
                </span>
              )
            }
          />
        </div>
      </div>

      {/* Body — Dispatch scrolls in its own container, Log handles its own. */}
      <div className="flex-grow min-h-0 flex flex-col">
        {tab === 'dispatch' ? (
          <div className="flex-grow min-h-0 px-4 pt-4 pb-4 overflow-y-auto overflow-x-hidden custom-scrollbar">
            <DispatchView telemetry={telemetry} loading={loading} />
          </div>
        ) : (
          <LogView logs={logs} loading={loading} />
        )}
      </div>
    </div>
  );
}
