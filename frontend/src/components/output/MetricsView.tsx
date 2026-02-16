"use client";

import { useTheme } from "next-themes";
import { getArtifactUrl } from "@/lib/api";

interface MetricsViewProps {
  preview: Record<string, unknown>;
  pipelineId: string;
  nodeId: string;
}

function formatMetric(value: number): string {
  if (Math.abs(value) < 0.01 || Math.abs(value) > 10000) {
    return value.toExponential(3);
  }
  return value.toFixed(4);
}

export function MetricsView({ preview, pipelineId, nodeId }: MetricsViewProps) {
  const { theme } = useTheme();
  const coefficients = preview.coefficients as
    | Record<string, number>
    | undefined;
  const intercept = preview.intercept as number | undefined;
  const chartRef = preview.chart_ref as string | undefined;

  // Support both wrapped format ({metrics: {...}}) and flat format ({accuracy: 0.96, ...})
  const knownNonMetricKeys = new Set(["coefficients", "intercept", "chart_ref", "gradient_norms"]);
  const metrics: Record<string, number> | undefined = preview.metrics
    ? (preview.metrics as Record<string, number>)
    : (() => {
        const flat: Record<string, number> = {};
        for (const [k, v] of Object.entries(preview)) {
          if (!knownNonMetricKeys.has(k) && (typeof v === "number" || typeof v === "string")) {
            flat[k] = typeof v === "number" ? v : Number(v);
          }
        }
        return Object.keys(flat).length > 0 ? flat : undefined;
      })();

  // Gradient norms from Backward Pass card
  const gradientNorms = preview.gradient_norms as Record<string, number> | undefined;
  
  const isDark = theme === "dark" || (theme === "system" && typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches);

  return (
    <div className={isDark ? "text-gray-200" : "text-gray-900"}>
      {metrics && Object.keys(metrics).length > 0 && (
        <div className="grid grid-cols-2 gap-1.5 mb-3">
          {Object.entries(metrics).map(([name, value]) => (
            <div
              key={name}
              className={`rounded-md border p-2 text-center ${
                isDark 
                  ? "bg-gray-800 border-gray-700" 
                  : "bg-white border-gray-300"
              }`}
            >
              <div className={`text-[10px] uppercase ${
                isDark ? "text-gray-400" : "text-gray-600"
              }`}>
                {name}
              </div>
              <div className={`text-sm font-semibold font-mono ${
                isDark ? "text-white" : "text-gray-900"
              }`}>
                {typeof value === "number" ? formatMetric(value) : String(value)}
              </div>
            </div>
          ))}
        </div>
      )}

      {coefficients && Object.keys(coefficients).length > 0 && (
        <div className="mb-3">
          <h4 className={`text-[10px] font-semibold uppercase mb-1 ${
            isDark ? "text-gray-400" : "text-gray-600"
          }`}>
            Coefficients
          </h4>
          <div className={`text-[10px] space-y-0.5 ${
            isDark ? "text-gray-300" : "text-gray-800"
          }`}>
            {Object.entries(coefficients).map(([feat, coef]) => (
              <div key={feat} className="flex justify-between">
                <span className="truncate mr-2">{feat}</span>
                <span className="font-mono shrink-0">{coef.toFixed(4)}</span>
              </div>
            ))}
            {intercept !== undefined && intercept !== null && (
              <div className={`flex justify-between border-t pt-0.5 mt-0.5 ${
                isDark ? "border-gray-700" : "border-gray-300"
              }`}>
                <span>intercept</span>
                <span className="font-mono">{intercept.toFixed(4)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {gradientNorms && Object.keys(gradientNorms).length > 0 && (
        <div className="mb-3">
          <h4 className={`text-[10px] font-semibold uppercase mb-1 ${
            isDark ? "text-gray-400" : "text-gray-600"
          }`}>
            Gradient Norms
          </h4>
          <div className={`text-[10px] space-y-0.5 ${
            isDark ? "text-gray-300" : "text-gray-800"
          }`}>
            {Object.entries(gradientNorms).map(([name, norm]) => (
              <div key={name} className="flex justify-between">
                <span className="truncate mr-2">{name}</span>
                <span className="font-mono shrink-0">{typeof norm === "number" ? norm.toFixed(6) : String(norm)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {chartRef && (
        <div>
          <h4 className={`text-[10px] font-semibold uppercase mb-1 ${
            isDark ? "text-gray-400" : "text-gray-600"
          }`}>
            Chart
          </h4>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={getArtifactUrl(pipelineId, nodeId, "eval_chart")}
            alt="Evaluation chart"
            className={`rounded border w-full ${
              isDark ? "border-gray-700" : "border-gray-300"
            }`}
          />
        </div>
      )}
    </div>
  );
}
