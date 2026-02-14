"use client";

import { useState } from "react";
import { useTheme } from "next-themes";

interface DataTableViewProps {
  preview: Record<string, unknown>;
}

function SimpleTable({
  rows,
  columns,
  isDark,
}: {
  rows: Record<string, unknown>[];
  columns: { name: string; dtype: string }[];
  isDark: boolean;
}) {
  if (!rows || !columns || columns.length === 0) return null;

  return (
    <div className={`overflow-auto max-h-64 border rounded ${isDark ? "border-gray-700" : "border-gray-300"}`}>
      <table className={`text-[10px] w-full ${isDark ? "text-gray-200" : "text-gray-900"}`}>
        <thead className={isDark ? "bg-gray-800 sticky top-0" : "bg-gray-200 sticky top-0"}>
          <tr>
            {columns.map((col) => (
              <th
                key={col.name}
                className={`px-2 py-1 text-left font-medium whitespace-nowrap ${isDark ? "text-white" : "text-gray-900"}`}
              >
                {col.name}
                <span className={`ml-1 font-normal ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                  {col.dtype}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr 
              key={i} 
              className={`border-t ${isDark ? "border-gray-700 hover:bg-gray-800/50" : "border-gray-300 hover:bg-gray-100"}`}
            >
              {columns.map((col) => (
                <td
                  key={col.name}
                  className={`px-2 py-0.5 truncate max-w-24 ${isDark ? "text-gray-300" : "text-gray-800"}`}
                >
                  {row[col.name] !== null && row[col.name] !== undefined
                    ? String(row[col.name])
                    : ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DataTableView({ preview }: DataTableViewProps) {
  const { theme } = useTheme();
  const [tab, setTab] = useState<"train" | "test">("train");
  
  const isDark = theme === "dark" || (theme === "system" && typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches);

  // Data split format
  if ("train" in preview && "test" in preview) {
    const train = preview.train as {
      rows: Record<string, unknown>[];
      row_count: number;
    };
    const test = preview.test as {
      rows: Record<string, unknown>[];
      row_count: number;
    };
    const ratio = preview.split_ratio as { train: number; test: number };

    const current = tab === "train" ? train : test;
    const cols = current.rows?.length
      ? Object.keys(current.rows[0]).map((name) => ({
          name,
          dtype: typeof current.rows[0][name] === "number" ? "number" : "string",
        }))
      : [];

    return (
      <div>
        {ratio && (
          <div className={`text-[10px] mb-2 ${isDark ? "text-gray-400" : "text-gray-600"}`}>
            Split: {(ratio.train * 100).toFixed(1)}% train / {(ratio.test * 100).toFixed(1)}% test
          </div>
        )}
        <div className="flex gap-1 mb-2">
          <button
            onClick={() => setTab("train")}
            className={`px-2 py-0.5 text-[10px] rounded transition-colors ${
              tab === "train" 
                ? "bg-blue-500 text-white" 
                : isDark
                  ? "bg-gray-700 text-gray-300 hover:bg-gray-600"
                  : "bg-gray-300 text-gray-700 hover:bg-gray-400"
            }`}
          >
            Train ({train.row_count})
          </button>
          <button
            onClick={() => setTab("test")}
            className={`px-2 py-0.5 text-[10px] rounded transition-colors ${
              tab === "test" 
                ? "bg-blue-500 text-white" 
                : isDark
                  ? "bg-gray-700 text-gray-300 hover:bg-gray-600"
                  : "bg-gray-300 text-gray-700 hover:bg-gray-400"
            }`}
          >
            Test ({test.row_count})
          </button>
        </div>
        <SimpleTable rows={current.rows || []} columns={cols} isDark={isDark} />
      </div>
    );
  }

  // Standard table format
  const rows = preview.rows as Record<string, unknown>[] | undefined;
  const columns = preview.columns as
    | { name: string; dtype: string }[]
    | undefined;
  const shape = preview.shape as { rows: number; cols: number } | undefined;
  const rowCount = preview.row_count as number | undefined;

  return (
    <div>
      <div className={`text-[10px] mb-2 ${isDark ? "text-gray-400" : "text-gray-600"}`}>
        {shape
          ? `${shape.rows.toLocaleString()} rows x ${shape.cols} columns`
          : rowCount
            ? `${rowCount.toLocaleString()} rows`
            : ""}
      </div>
      {rows && columns && <SimpleTable rows={rows} columns={columns} isDark={isDark} />}
    </div>
  );
}
