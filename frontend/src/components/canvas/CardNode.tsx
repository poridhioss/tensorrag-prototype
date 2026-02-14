"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import type { CardNodeData } from "@/lib/types";
import { usePipelineStore } from "@/store/pipelineStore";
import { runPipeline } from "@/lib/pipelineRunner";

const CATEGORY_STYLES: Record<string, string> = {
  data: "border-card-data/60 bg-card-data/5",
  model: "border-card-model/60 bg-card-model/5",
  evaluation: "border-card-evaluation/60 bg-card-evaluation/5",
  inference: "border-card-inference/60 bg-card-inference/5",
};

const STATUS_DOT: Record<string, string> = {
  pending: "bg-status-pending",
  running: "bg-status-running animate-pulse",
  completed: "bg-status-completed",
  failed: "bg-status-failed",
};

function compactPreview(
  viewType: string,
  preview: Record<string, unknown>
): string {
  if (viewType === "table") {
    const shape = preview.shape as { rows: number; cols: number } | undefined;
    if (shape) return `${shape.rows.toLocaleString()} rows x ${shape.cols} cols`;
    const rowCount = preview.row_count as number | undefined;
    if (rowCount) return `${rowCount.toLocaleString()} rows`;
    const train = preview.train as { row_count: number } | undefined;
    const test = preview.test as { row_count: number } | undefined;
    if (train && test) return `Train: ${train.row_count} / Test: ${test.row_count}`;
    return "Table";
  }
  if (viewType === "metrics") {
    const metrics = preview.metrics as Record<string, number> | undefined;
    if (metrics) {
      const entries = Object.entries(metrics);
      if (entries.length > 0) {
        const [name, val] = entries[0];
        return `${name}: ${typeof val === "number" ? val.toFixed(4) : val}`;
      }
    }
    return "Metrics";
  }
  if (viewType === "model_summary") {
    const mt = preview.model_type as string | undefined;
    return mt || "Model";
  }
  return "";
}

export const CardNode = memo(function CardNode({
  id,
  data,
  selected,
}: NodeProps<Node<CardNodeData>>) {
  const { cardSchema, status, outputPreview } = data;
  const isExecuting = usePipelineStore((s) => s.isExecuting);
  const logEntries = usePipelineStore((s) => s.logEntries);
  const setConsoleOpen = usePipelineStore((s) => s.setConsoleOpen);
  const setConsoleActiveTab = usePipelineStore((s) => s.setConsoleActiveTab);
  const setSelectedNodeIdForOutput = usePipelineStore((s) => s.setSelectedNodeIdForOutput);
  const removeNode = usePipelineStore((s) => s.removeNode);

  const inputKeys = Object.keys(cardSchema.input_schema);
  const outputKeys = Object.keys(cardSchema.output_schema);

  const showOutput = status === "completed" || status === "failed";
  const isRunning = status === "running";
  const hasLogs = isExecuting || logEntries.length > 0;

  return (
    <>
      <div
        className={`group relative rounded-lg border shadow-sm transition-all bg-white w-72
          ${CATEGORY_STYLES[cardSchema.category] || "border-border"}
          ${selected ? "ring-2 ring-accent shadow-lg border-accent" : "hover:shadow-md"}`}
        style={{ minHeight: `${Math.max(90, 50 + (inputKeys.length + outputKeys.length) * 8)}px` }}
      >
        {/* Input handles with labels - label ABOVE handle (outside card), handle ON border */}
        {inputKeys.map((key, i) => {
          const spacing = 100 / (inputKeys.length + 1);
          const topPercent = spacing * (i + 1);
          return (
            <div 
              key={`in-${key}`} 
              className="absolute left-0 z-10 flex flex-col items-center" 
              style={{ top: `${topPercent}%`, transform: "translateY(-50%)" }}
            >
              <div 
                className="text-[9px] text-text-secondary/90 font-medium pointer-events-none whitespace-nowrap absolute" 
                style={{ bottom: "100%", marginBottom: "0.25rem", left: "-2rem" }} 
                title={key}
              >
                {key}
              </div>
              <Handle
                type="target"
                position={Position.Left}
                id={key}
                className="!w-2.5 !h-2.5 !bg-accent !border-2 !border-background hover:!bg-accent/80 transition-colors"
              />
            </div>
          );
        })}

        {/* Delete button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (confirm("Remove this card?")) {
              removeNode(id);
            }
          }}
          className="absolute -top-2 -right-2 z-20
            w-5 h-5 flex items-center justify-center
            bg-background border border-border rounded-full
            text-text-secondary/60 hover:text-status-failed 
            hover:bg-status-failed/10 hover:border-status-failed/30
            transition-all shadow-sm
            opacity-0 group-hover:opacity-100"
          title="Remove card"
        >
          <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Card content */}
        <div className="px-4 py-1.5 h-full flex flex-col">
          {/* Title section - centered */}
          <div className="text-center mb-1">
            <div className="flex items-center justify-center gap-2 mb-0">
              <h3 className="text-xs font-semibold text-text-primary break-words">
                {cardSchema.display_name}
              </h3>
              <div className="flex items-center gap-1.5 shrink-0">
                {isRunning && (
                  <span className="w-2 h-2 rounded-full bg-status-running animate-pulse" />
                )}
                {!isExecuting && !isRunning && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      runPipeline(id);
                    }}
                    className="text-text-secondary/60 hover:text-accent transition-colors p-0.5"
                    title="Run from this node"
                  >
                    <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  </button>
                )}
                {!isRunning && (
                  <span
                    className={`w-2 h-2 rounded-full ${STATUS_DOT[status]}`}
                  />
                )}
              </div>
            </div>
          </div>

          {/* Description section - centered, no box */}
          {cardSchema.description && (
            <div className="text-center mb-1.5">
              <p className="text-[10px] text-text-secondary/80 leading-relaxed break-words">
                {cardSchema.description}
              </p>
            </div>
          )}

          {/* Status info */}
          {status === "completed" && outputPreview && (
            <div className="text-center mb-2 text-[10px] text-text-secondary/80 truncate">
              {compactPreview(cardSchema.output_view_type, outputPreview.preview)}
            </div>
          )}

          {status === "failed" && data.error && (
            <div className="text-center mb-2 text-[10px] text-status-failed truncate">
              Error: {data.error.substring(0, 50)}
            </div>
          )}

          {/* Output and Console section - bottom, centered */}
          <div className="mt-auto pt-1 flex items-center justify-center gap-3">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setSelectedNodeIdForOutput(id);
                setConsoleActiveTab("output");
                setConsoleOpen(true);
              }}
              className="inline-flex items-center gap-0.5 text-[9px] text-text-secondary/70 hover:text-text-primary transition-colors px-1.5 py-0.5"
            >
              <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              Output
            </button>
            {(hasLogs || isExecuting) && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setConsoleOpen(true);
                }}
                className="inline-flex items-center gap-0.5 text-[9px] text-text-secondary/70 hover:text-text-primary transition-colors px-1.5 py-0.5"
                title="Open console"
              >
                <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                Console
              </button>
            )}
          </div>
        </div>

        {/* Output handles with labels - label ABOVE handle (outside card), handle ON border */}
        {outputKeys.map((key, i) => {
          const spacing = 100 / (outputKeys.length + 1);
          const topPercent = spacing * (i + 1);
          return (
            <div 
              key={`out-${key}`} 
              className="absolute right-0 z-10 flex flex-col items-center" 
              style={{ top: `${topPercent}%`, transform: "translateY(-50%)" }}
            >
              <div 
                className="text-[9px] text-text-secondary/90 font-medium pointer-events-none whitespace-nowrap absolute" 
                style={{ bottom: "100%", marginBottom: "0.25rem", right: "-2rem" }} 
                title={key}
              >
                {key}
              </div>
              <Handle
                type="source"
                position={Position.Right}
                id={key}
                className="!w-2.5 !h-2.5 !bg-accent !border-2 !border-background hover:!bg-accent/80 transition-colors"
              />
            </div>
          );
        })}
      </div>

    </>
  );
});
