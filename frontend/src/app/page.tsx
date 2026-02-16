"use client";

import { useEffect, useState, useRef } from "react";
import dynamic from "next/dynamic";
import { ReactFlowProvider } from "@xyflow/react";
import { usePipelineStore } from "@/store/pipelineStore";
import { useWorkspaceStore } from "@/store/workspaceStore";
import { Header } from "@/components/header/Header";
import { CardPalette } from "@/components/sidebar/CardPalette";
import { PipelineCanvas } from "@/components/canvas/PipelineCanvas";
import { ConsolePanel } from "@/components/console/ConsolePanel";

const CardEditorView = dynamic(
  () =>
    import("@/components/editor/CardEditorView").then((mod) => ({
      default: mod.CardEditorView,
    })),
  {
    ssr: false,
    loading: () => (
      <div className="flex-1 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent" />
      </div>
    ),
  }
);

const PALETTE_MIN = 180;
const PALETTE_MAX = 400;
const PALETTE_DEFAULT = 224;

function PipelineApp() {
  const validationErrors = usePipelineStore((s) => s.validationErrors);
  const setValidationErrors = usePipelineStore((s) => s.setValidationErrors);
  const activeView = usePipelineStore((s) => s.activeView);

  const loadProjects = useWorkspaceStore((s) => s.loadProjects);
  const switchProject = useWorkspaceStore((s) => s.switchProject);

  // Resizable palette
  const [paletteWidth, setPaletteWidth] = useState(PALETTE_DEFAULT);
  const [isResizing, setIsResizing] = useState(false);
  const resizeStartX = useRef(0);
  const resizeStartWidth = useRef(0);

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      const delta = e.clientX - resizeStartX.current;
      setPaletteWidth(Math.min(PALETTE_MAX, Math.max(PALETTE_MIN, resizeStartWidth.current + delta)));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isResizing]);

  // Load projects on mount and restore active project
  useEffect(() => {
    async function init() {
      await loadProjects();

      // If there was a previously active project (persisted in localStorage),
      // re-activate it to load its cards and pipeline
      const persisted = useWorkspaceStore.getState().activeProject;
      if (persisted) {
        try {
          await switchProject(persisted);
        } catch {
          // Project may have been deleted — clear it
          useWorkspaceStore.getState().setActiveProject(null);
        }
      }
    }
    init();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="h-screen flex flex-col">
      <Header />

      {activeView === "board" ? (
        <>
          {validationErrors.length > 0 && (
            <div className="bg-status-failed/10 border-b border-status-failed/30 px-4 py-2 flex items-start gap-2">
              <div className="flex-1">
                {validationErrors.map((err, i) => (
                  <p key={i} className="text-xs text-status-failed">
                    {err}
                  </p>
                ))}
              </div>
              <button
                onClick={() => setValidationErrors([])}
                className="text-xs text-status-failed hover:text-status-failed/70"
              >
                Dismiss
              </button>
            </div>
          )}

          <div className="flex-1 flex overflow-hidden relative">
            <div
              className="shrink-0 overflow-hidden"
              style={{ width: `${paletteWidth}px` }}
            >
              <CardPalette />
            </div>

            {/* Resize handle */}
            <div
              onMouseDown={(e) => {
                e.preventDefault();
                resizeStartX.current = e.clientX;
                resizeStartWidth.current = paletteWidth;
                setIsResizing(true);
              }}
              className={`w-1 shrink-0 cursor-col-resize transition-colors relative group
                ${isResizing ? "bg-accent/50" : "hover:bg-accent/30"}`}
            >
              <div className="absolute inset-y-0 -left-1 -right-1" />
            </div>

            <div className="flex-1 flex flex-col overflow-hidden relative">
              <PipelineCanvas />
              <ConsolePanel />
            </div>
          </div>
        </>
      ) : (
        <CardEditorView />
      )}
    </div>
  );
}

export default function Home() {
  return (
    <ReactFlowProvider>
      <PipelineApp />
    </ReactFlowProvider>
  );
}
