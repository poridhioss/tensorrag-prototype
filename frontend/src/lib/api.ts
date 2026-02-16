import type {
  CardSchema,
  PipelineRequest,
  PipelineStatus,
  CardOutputPreview,
  CardValidationResult,
  CustomCardFile,
  CardFileEntry,
  SavedPipelineState,
  RegisteredCardInfo,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchCards(): Promise<CardSchema[]> {
  return fetchJSON<CardSchema[]>(`${API_URL}/api/cards`);
}

export async function validatePipeline(
  pipeline: PipelineRequest
): Promise<{ errors: string[] }> {
  return fetchJSON(`${API_URL}/api/pipeline/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(pipeline),
  });
}

export async function executePipeline(
  pipeline: PipelineRequest
): Promise<{ pipeline_id: string; status: string }> {
  return fetchJSON(`${API_URL}/api/pipeline/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(pipeline),
  });
}

export async function fetchPipelineStatus(
  pipelineId: string
): Promise<PipelineStatus> {
  return fetchJSON(`${API_URL}/api/pipeline/${pipelineId}/status`);
}

export async function fetchCardOutput(
  pipelineId: string,
  nodeId: string
): Promise<CardOutputPreview> {
  return fetchJSON(`${API_URL}/api/card/${pipelineId}/${nodeId}/output`);
}

export function getArtifactUrl(
  pipelineId: string,
  nodeId: string,
  key: string
): string {
  return `${API_URL}/api/artifacts/${pipelineId}/${nodeId}/${key}`;
}

// --- Custom card editor API ---

export async function validateCardCode(
  sourceCode: string
): Promise<CardValidationResult> {
  return fetchJSON<CardValidationResult>(`${API_URL}/api/cards/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_code: sourceCode }),
  });
}

export async function uploadCustomCard(
  filename: string,
  sourceCode: string
): Promise<{ success: boolean; card_type: string }> {
  return fetchJSON(`${API_URL}/api/cards/custom`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, source_code: sourceCode }),
  });
}

export async function listCustomCards(): Promise<CustomCardFile[]> {
  return fetchJSON<CustomCardFile[]>(`${API_URL}/api/cards/custom`);
}

export async function deleteCustomCard(
  cardType: string
): Promise<{ success: boolean }> {
  return fetchJSON(`${API_URL}/api/cards/custom/${encodeURIComponent(cardType)}`, {
    method: "DELETE",
  });
}

// --- Workspace API ---

const WS_URL = `${API_URL}/api/workspace`;

export async function listProjects(): Promise<string[]> {
  return fetchJSON<string[]>(`${WS_URL}/projects`);
}

export async function createProject(
  name: string
): Promise<{ success: boolean; name: string }> {
  return fetchJSON(`${WS_URL}/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function deleteProject(name: string): Promise<{ success: boolean }> {
  return fetchJSON(`${WS_URL}/projects/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export async function loadPipelineState(
  project: string
): Promise<SavedPipelineState> {
  return fetchJSON<SavedPipelineState>(
    `${WS_URL}/projects/${encodeURIComponent(project)}/pipeline`
  );
}

export async function savePipelineState(
  project: string,
  state: SavedPipelineState
): Promise<void> {
  await fetchJSON(`${WS_URL}/projects/${encodeURIComponent(project)}/pipeline`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
  });
}

export async function listProjectCards(
  project: string
): Promise<CardFileEntry[]> {
  return fetchJSON<CardFileEntry[]>(
    `${WS_URL}/projects/${encodeURIComponent(project)}/cards`
  );
}

export async function getCardSource(
  project: string,
  path: string
): Promise<{ path: string; source_code: string }> {
  return fetchJSON(
    `${WS_URL}/projects/${encodeURIComponent(project)}/cards/source`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }
  );
}

export async function saveProjectCard(
  project: string,
  path: string,
  sourceCode: string
): Promise<{ success: boolean; card_type: string }> {
  return fetchJSON(
    `${WS_URL}/projects/${encodeURIComponent(project)}/cards`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, source_code: sourceCode }),
    }
  );
}

export async function deleteProjectCard(
  project: string,
  path: string
): Promise<{ success: boolean }> {
  return fetchJSON(
    `${WS_URL}/projects/${encodeURIComponent(project)}/cards`,
    {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }
  );
}

export async function createProjectFolder(
  project: string,
  path: string
): Promise<{ success: boolean }> {
  return fetchJSON(
    `${WS_URL}/projects/${encodeURIComponent(project)}/cards/folder`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }
  );
}

export async function deleteProjectFolder(
  project: string,
  path: string
): Promise<{ success: boolean }> {
  return fetchJSON(
    `${WS_URL}/projects/${encodeURIComponent(project)}/cards/folder`,
    {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }
  );
}

export async function activateProject(
  project: string
): Promise<{ registered: RegisteredCardInfo[]; cards: CardSchema[] }> {
  return fetchJSON(
    `${WS_URL}/projects/${encodeURIComponent(project)}/activate`,
    { method: "POST" }
  );
}
