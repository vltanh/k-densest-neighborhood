import { useState, useEffect } from 'react';

export function useDragResize() {
  const [sidebarWidth, setSidebarWidth] = useState(460);
  const [isDraggingSidebar, setIsDraggingSidebar] = useState(false);
  const [ledgerHeightPct, setLedgerHeightPct] = useState(40);
  const [isDraggingLedger, setIsDraggingLedger] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (isDraggingSidebar) {
        let w = e.clientX;
        if (w < 150) w = 0;
        if (w > 800) w = 800;
        setSidebarWidth(w);
      }
      if (isDraggingLedger) {
        let h = ((window.innerHeight - e.clientY) / window.innerHeight) * 100;
        if (h < 5) h = 0;
        if (h > 90) h = 90;
        setLedgerHeightPct(h);
      }
    };
    const handleMouseUp = () => {
      setIsDraggingSidebar(false);
      setIsDraggingLedger(false);
    };
    if (isDraggingSidebar || isDraggingLedger) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.userSelect = 'none';
    } else {
      document.body.style.userSelect = 'auto';
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.userSelect = 'auto';
    };
  }, [isDraggingSidebar, isDraggingLedger]);

  return {
    sidebarWidth, setSidebarWidth, setIsDraggingSidebar,
    ledgerHeightPct, setLedgerHeightPct, setIsDraggingLedger,
  };
}
