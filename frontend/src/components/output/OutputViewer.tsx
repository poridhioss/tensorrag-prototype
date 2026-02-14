"use client";

import { useTheme } from "next-themes";
import { DataTableView } from "./DataTableView";
import { MetricsView } from "./MetricsView";
import { ModelSummaryView } from "./ModelSummaryView";

interface OutputViewerProps {
  outputType: string;
  preview: Record<string, unknown>;
  pipelineId: string;
  nodeId: string;
}

export function OutputViewer({
  outputType,
  preview,
  pipelineId,
  nodeId,
}: OutputViewerProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark" || (theme === "system" && typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  
  switch (outputType) {
    case "table":
      return <DataTableView preview={preview} />;
    case "metrics":
      return (
        <MetricsView
          preview={preview}
          pipelineId={pipelineId}
          nodeId={nodeId}
        />
      );
    case "model_summary":
      return <ModelSummaryView preview={preview} />;
    default:
      return (
        <pre className={`text-[10px] overflow-auto max-h-48 p-2 rounded border ${
          isDark 
            ? "bg-gray-800 border-gray-700 text-gray-200"
            : "bg-white border-gray-300 text-gray-900"
        }`}>
          {JSON.stringify(preview, null, 2)}
        </pre>
      );
  }
}
