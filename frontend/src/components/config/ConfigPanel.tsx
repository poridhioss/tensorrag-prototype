"use client";

import { usePipelineStore } from "@/store/pipelineStore";
import { SchemaForm } from "./SchemaForm";
import { OutputViewer } from "@/components/output/OutputViewer";

export function ConfigPanel() {
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId);
  const nodes = usePipelineStore((s) => s.nodes);
  const updateNodeConfig = usePipelineStore((s) => s.updateNodeConfig);
  const pipelineId = usePipelineStore((s) => s.pipelineId);

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const isOpen = !!selectedNode;

  return (
    <div
      className={`border-l border-border bg-bg-secondary overflow-y-auto transition-all duration-200 shrink-0
        ${isOpen ? "w-80" : "w-0 border-0"}`}
    >
      {selectedNode && (
        <div className="p-4 w-80">
          <div className="flex items-center justify-between mb-1">
            <h2 className="text-sm font-semibold">
              {selectedNode.data.cardSchema.display_name}
            </h2>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-border text-text-secondary">
              {selectedNode.data.cardSchema.category}
            </span>
          </div>
          <p className="text-xs text-text-secondary mb-4">
            {selectedNode.data.cardSchema.description}
          </p>

          <h3 className="text-xs font-semibold uppercase text-text-secondary mb-2">
            Configuration
          </h3>
          <SchemaForm
            schema={selectedNode.data.cardSchema.config_schema}
            values={selectedNode.data.config}
            onChange={(config) => updateNodeConfig(selectedNode.id, config)}
          />

          {selectedNode.data.status === "failed" && selectedNode.data.error && (
            <div className="mt-4 p-2 rounded-md bg-status-failed/10 border border-status-failed/30">
              <h3 className="text-xs font-semibold text-status-failed mb-1">
                Error
              </h3>
              <p className="text-[10px] text-status-failed">
                {selectedNode.data.error}
              </p>
            </div>
          )}

          {selectedNode.data.status === "completed" &&
            selectedNode.data.outputPreview && (
              <div className="mt-4 pt-4 border-t border-border">
                <h3 className="text-xs font-semibold uppercase text-text-secondary mb-2">
                  Output
                </h3>
                <OutputViewer
                  outputType={selectedNode.data.outputPreview.output_type}
                  preview={selectedNode.data.outputPreview.preview}
                  pipelineId={pipelineId || ""}
                  nodeId={selectedNode.id}
                />
              </div>
            )}
        </div>
      )}
    </div>
  );
}
