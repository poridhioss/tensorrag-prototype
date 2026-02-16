import { usePipelineStore } from "@/store/pipelineStore";
import { useWorkspaceStore } from "@/store/workspaceStore";
import { validatePipeline, executePipeline, fetchCardOutput } from "@/lib/api";
import { connectWebSocket } from "@/hooks/useWebSocket";
import type { WSMessage } from "@/lib/types";

let ws: WebSocket | null = null;

function onMessage(msg: WSMessage) {
  const state = usePipelineStore.getState();

  if (msg.type === "log") {
    state.addLog(msg.text);
    return;
  }

  // node_status
  state.setNodeStatus(msg.node_id, msg.status, msg.message || null);

  if (msg.status === "completed" && state.pipelineId) {
    fetchCardOutput(state.pipelineId, msg.node_id)
      .then((output) => {
        const s = usePipelineStore.getState();
        s.setNodeOutput(msg.node_id, output);
        // Auto-switch to output tab when the selected node completes
        if (s.selectedNodeIdForOutput === msg.node_id) {
          s.setConsoleActiveTab("output");
        }
      })
      .catch(() => {});
  }

  const updated = usePipelineStore.getState();
  const execIds = new Set(updated.executingNodeIds);
  const allDone = updated.nodes
    .filter((n) => execIds.has(n.id))
    .every((n) => n.data.status === "completed" || n.data.status === "failed");
  if (allDone && updated.isExecuting) {
    updated.setIsExecuting(false);
    ws?.close();
    ws = null;
  }
}

export async function runPipeline(targetNodeId?: string) {
  const store = usePipelineStore.getState();
  if (store.isExecuting || store.nodes.length === 0) return;

  // Use project-scoped pipeline ID (double-underscore separator is URL-safe)
  const activeProject = useWorkspaceStore.getState().activeProject;
  const pid = activeProject
    ? `workspace__${activeProject}`
    : store.pipelineId || `pipeline-${Date.now()}`;

  // If targeting a specific node, only run it + its upstream dependencies
  const nodeIds = targetNodeId
    ? store.getUpstreamNodeIds(targetNodeId)
    : store.nodes.map((n) => n.id);

  const request = store.toPipelineRequest(pid, nodeIds);

  store.setValidationErrors([]);
  store.clearLogs();

  try {
    const { errors } = await validatePipeline(request);
    if (errors.length > 0) {
      store.setValidationErrors(errors);
      return;
    }
  } catch (err) {
    store.setValidationErrors([
      err instanceof Error ? err.message : "Validation failed",
    ]);
    return;
  }

  store.resetNodeStatuses(nodeIds);
  store.setExecutingNodeIds(nodeIds);
  store.setPipelineId(pid);
  store.setIsExecuting(true);
  store.setConsoleOpen(true);
  store.setConsoleActiveTab("console");

  // Select the target node for output viewing
  if (targetNodeId) {
    store.setSelectedNodeIdForOutput(targetNodeId);
  }

  try {
    if (ws) ws.close();
    ws = await connectWebSocket(pid, onMessage);
  } catch {
    store.setValidationErrors(["WebSocket connection failed"]);
    store.setIsExecuting(false);
    return;
  }

  try {
    await executePipeline(request);
  } catch (err) {
    store.setValidationErrors([
      err instanceof Error ? err.message : "Execution failed",
    ]);
    store.setIsExecuting(false);
    ws?.close();
    ws = null;
  }
}
