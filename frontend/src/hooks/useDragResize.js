import { useState, useEffect } from 'react';

const initialSidebarWidth = () => {
  if (typeof window === 'undefined') return 460;
  const vw = window.innerWidth || 1024;
  if (vw < 768) return 0;                    // mobile: collapsed by default
  return Math.min(460, Math.floor(vw * 0.85));
};

export function useDragResize() {
  const [sidebarWidth, setSidebarWidth] = useState(initialSidebarWidth);
  const [isDraggingSidebar, setIsDraggingSidebar] = useState(false);
  const [ledgerHeightPct, setLedgerHeightPct] = useState(40);
  const [isDraggingLedger, setIsDraggingLedger] = useState(false);

  useEffect(() => {
    const pointFromEvent = (e) => {
      if (e.touches && e.touches[0]) return { x: e.touches[0].clientX, y: e.touches[0].clientY };
      return { x: e.clientX, y: e.clientY };
    };
    const handleMove = (e) => {
      const { x, y } = pointFromEvent(e);
      if (isDraggingSidebar) {
        const vw = window.innerWidth;
        let w = x;
        if (w < 150) w = 0;
        const max = Math.min(800, vw - 40);
        if (w > max) w = max;
        setSidebarWidth(w);
      }
      if (isDraggingLedger) {
        let h = ((window.innerHeight - y) / window.innerHeight) * 100;
        if (h < 5) h = 0;
        if (h > 90) h = 90;
        setLedgerHeightPct(h);
      }
    };
    const handleUp = () => {
      setIsDraggingSidebar(false);
      setIsDraggingLedger(false);
    };
    if (isDraggingSidebar || isDraggingLedger) {
      document.addEventListener('mousemove', handleMove);
      document.addEventListener('mouseup', handleUp);
      document.addEventListener('touchmove', handleMove, { passive: false });
      document.addEventListener('touchend', handleUp);
      document.body.style.userSelect = 'none';
    } else {
      document.body.style.userSelect = 'auto';
    }
    return () => {
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleUp);
      document.removeEventListener('touchmove', handleMove);
      document.removeEventListener('touchend', handleUp);
      document.body.style.userSelect = 'auto';
    };
  }, [isDraggingSidebar, isDraggingLedger]);

  return {
    sidebarWidth, setSidebarWidth, setIsDraggingSidebar,
    ledgerHeightPct, setLedgerHeightPct, setIsDraggingLedger,
  };
}
