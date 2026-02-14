from pydantic import BaseModel


class NodeConfig(BaseModel):
    id: str
    type: str
    config: dict = {}
    position: dict = {"x": 0, "y": 0}


class EdgeConfig(BaseModel):
    source: str
    target: str
    source_output: str = "default"
    target_input: str = "default"


class PipelineRequest(BaseModel):
    pipeline_id: str
    nodes: list[NodeConfig]
    edges: list[EdgeConfig]


class NodeStatus(BaseModel):
    node_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    error: str | None = None


class PipelineStatus(BaseModel):
    pipeline_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    node_statuses: dict[str, NodeStatus]


class CardOutputPreview(BaseModel):
    node_id: str
    output_type: str
    preview: dict
