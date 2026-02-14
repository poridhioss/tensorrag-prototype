"use client";

import { useEffect } from "react";
import { createPortal } from "react-dom";
import { usePipelineStore } from "@/store/pipelineStore";
import { OutputViewer } from "./OutputViewer";
import type { CardNodeData } from "@/lib/types";

interface OutputModalProps {
  nodeData: CardNodeData;
  nodeId: string;
  onClose: () => void;
}

export function OutputModal({ nodeData, nodeId, onClose }: OutputModalProps) {
  const pipelineId = usePipelineStore((s) => s.pipelineId);
  const { cardSchema, status, error, outputPreview } = nodeData;

  // Close on Escape key
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <div
        className="bg-background border border-border rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold">{cardSchema.display_name}</h2>
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium
              ${status === "running" ? "bg-status-running/20 text-status-running" : ""}
              ${status === "completed" ? "bg-status-completed/20 text-status-completed" : ""}
              ${status === "failed" ? "bg-status-failed/20 text-status-failed" : ""}
              ${status === "pending" ? "bg-status-pending/20 text-status-pending" : ""}
            `}>
              {status}
            </span>
          </div>
          <button
            onClick={onClose}
            className="w-6 h-6 flex items-center justify-center rounded hover:bg-bg-secondary text-text-secondary"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-4">
          {status === "running" && (
            <div className="flex flex-col items-center justify-center py-12 gap-3">
              <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <p className="text-sm text-text-secondary">Executing...</p>
            </div>
          )}

          {status === "pending" && (
            <div className="flex flex-col items-center justify-center py-12">
              <p className="text-sm text-text-secondary">Waiting to execute</p>
            </div>
          )}

          {status === "failed" && error && (
            <div className="p-3 rounded-md bg-status-failed/10 border border-status-failed/30">
              <h3 className="text-xs font-semibold text-status-failed mb-1">Error</h3>
              <p className="text-xs text-status-failed whitespace-pre-wrap">{error}</p>
            </div>
          )}

          {status === "completed" && outputPreview && (
            <OutputViewer
              outputType={outputPreview.output_type}
              preview={outputPreview.preview}
              pipelineId={pipelineId || ""}
              nodeId={nodeId}
            />
          )}

          {status === "completed" && !outputPreview && (
            <div className="flex flex-col items-center justify-center py-12">
              <p className="text-sm text-text-secondary">No output preview available</p>
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
}
