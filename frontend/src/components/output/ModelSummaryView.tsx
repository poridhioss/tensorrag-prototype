"use client";

import { useTheme } from "next-themes";

interface ModelSummaryViewProps {
  preview: Record<string, unknown>;
}

export function ModelSummaryView({ preview }: ModelSummaryViewProps) {
  const { theme } = useTheme();
  const modelType = preview.model_type as string | undefined;
  const hyperparameters = preview.hyperparameters as
    | Record<string, unknown>
    | undefined;
  
  const isDark = theme === "dark" || (theme === "system" && typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches);

  return (
    <div className={isDark ? "text-gray-200" : "text-gray-900"}>
      {modelType && (
        <div className="mb-2">
          <span className={`text-[10px] uppercase ${
            isDark ? "text-gray-400" : "text-gray-600"
          }`}>
            Model Type
          </span>
          <div className={`text-xs font-semibold ${
            isDark ? "text-white" : "text-gray-900"
          }`}>{modelType}</div>
        </div>
      )}
      {hyperparameters && Object.keys(hyperparameters).length > 0 && (
        <div>
          <span className={`text-[10px] uppercase ${
            isDark ? "text-gray-400" : "text-gray-600"
          }`}>
            Hyperparameters
          </span>
          <div className={`text-[10px] mt-1 space-y-0.5 ${
            isDark ? "text-gray-300" : "text-gray-800"
          }`}>
            {Object.entries(hyperparameters).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <span>{key}</span>
                <span className="font-mono">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {(!hyperparameters || Object.keys(hyperparameters).length === 0) &&
        modelType && (
          <p className={`text-[10px] ${
            isDark ? "text-gray-400" : "text-gray-600"
          }`}>
            Using default hyperparameters
          </p>
        )}
    </div>
  );
}
