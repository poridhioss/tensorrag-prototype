from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.config import settings
from app.services.executor import get_pipeline_state
from cards.registry import get_card

router = APIRouter(tags=["artifacts"])


def _get_storage():
    if settings.S3_ENABLED:
        from app.services.s3_storage import S3StorageService
        return S3StorageService()
    from app.services.storage import StorageService
    return StorageService()


@router.get("/card/{pipeline_id}/{node_id}/output")
def get_card_output(pipeline_id: str, node_id: str):
    state = get_pipeline_state(pipeline_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if node_id not in state.node_outputs:
        raise HTTPException(
            status_code=404,
            detail=f"No output for node '{node_id}'",
        )

    if state.node_statuses.get(node_id) != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Node '{node_id}' has not completed successfully",
        )

    outputs = state.node_outputs[node_id]
    storage = _get_storage()

    # Try each card to find which one matches this output structure
    from cards.registry import CARD_REGISTRY

    for card in CARD_REGISTRY.values():
        output_keys = set(card.output_schema.keys())
        if output_keys and output_keys.issubset(set(outputs.keys())):
            try:
                preview = card.get_output_preview(outputs, storage)
                return {
                    "node_id": node_id,
                    "output_type": card.output_view_type,
                    "preview": preview,
                }
            except Exception:
                continue

    raise HTTPException(
        status_code=500,
        detail=f"Could not generate preview for node '{node_id}'",
    )


@router.get("/artifacts/{pipeline_id}/{node_id}/{key}")
def get_artifact(pipeline_id: str, node_id: str, key: str):
    state = get_pipeline_state(pipeline_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    outputs = state.node_outputs.get(node_id, {})
    ref = outputs.get(key)
    if ref is None:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{key}' not found for node '{node_id}'",
        )

    # S3 refs start with "s3://"
    if ref.startswith("s3://"):
        storage = _get_storage()
        buf, suffix = storage.get_bytes_streaming(ref)
        media_types = {
            ".parquet": "application/octet-stream",
            ".json": "application/json",
            ".joblib": "application/octet-stream",
            ".png": "image/png",
            ".csv": "text/csv",
        }
        media_type = media_types.get(suffix, "application/octet-stream")
        filename = f"{key}{suffix}"
        return StreamingResponse(
            buf,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    # Local file path
    path = Path(ref)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    media_types = {
        ".parquet": "application/octet-stream",
        ".json": "application/json",
        ".joblib": "application/octet-stream",
        ".png": "image/png",
        ".csv": "text/csv",
    }
    media_type = media_types.get(path.suffix, "application/octet-stream")

    return FileResponse(path, media_type=media_type, filename=path.name)
