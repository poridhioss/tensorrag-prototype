from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.workspace_manager import workspace_manager
from app.services.card_validator import validate_card_source
from cards.registry import list_cards

router = APIRouter(tags=["workspace"])


# -- Request / Response models --

class CreateProjectRequest(BaseModel):
    name: str


class GetCardSourceRequest(BaseModel):
    path: str


class SaveCardRequest(BaseModel):
    path: str
    source_code: str


class DeleteCardRequest(BaseModel):
    path: str


class CreateFolderRequest(BaseModel):
    path: str


class PipelineStateRequest(BaseModel):
    nodes: list[dict]
    edges: list[dict]
    nodeCounter: int = 0


# -- Project CRUD --

@router.get("/projects")
def list_projects():
    return workspace_manager.list_projects()


@router.post("/projects")
def create_project(req: CreateProjectRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")
    existing = workspace_manager.list_projects()
    if name in existing:
        raise HTTPException(status_code=409, detail=f"Project '{name}' already exists")
    workspace_manager.create_project(name)
    return {"success": True, "name": name}


@router.delete("/projects/{name}")
def delete_project(name: str):
    workspace_manager.delete_project(name)
    return {"success": True}


# -- Pipeline state --

@router.get("/projects/{name}/pipeline")
def get_pipeline_state(name: str):
    return workspace_manager.load_pipeline_state(name)


@router.put("/projects/{name}/pipeline")
def save_pipeline_state(name: str, req: PipelineStateRequest):
    workspace_manager.save_pipeline_state(name, req.model_dump())
    return {"success": True}


# -- Card files --

@router.get("/projects/{name}/cards")
def list_card_files(name: str):
    return workspace_manager.list_card_files(name)


@router.post("/projects/{name}/cards/source")
def get_card_source(name: str, req: GetCardSourceRequest):
    try:
        source = workspace_manager.get_card_source(name, req.path)
        return {"path": req.path, "source_code": source}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/projects/{name}/cards")
def save_card_file(name: str, req: SaveCardRequest):
    # Validate first
    result = validate_card_source(req.source_code)
    if not result["success"]:
        raise HTTPException(status_code=400, detail={"errors": result["errors"]})

    try:
        card_type = workspace_manager.save_card_file(name, req.path, req.source_code)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"success": True, "card_type": card_type}


@router.delete("/projects/{name}/cards")
def delete_card_file(name: str, req: DeleteCardRequest):
    workspace_manager.delete_card_file(name, req.path)
    return {"success": True}


@router.post("/projects/{name}/cards/folder")
def create_folder(name: str, req: CreateFolderRequest):
    workspace_manager.create_folder(name, req.path)
    return {"success": True}


@router.delete("/projects/{name}/cards/folder")
def delete_folder(name: str, req: DeleteCardRequest):
    workspace_manager.delete_folder(name, req.path)
    return {"success": True}


# -- Activate project (load & register cards) --

@router.post("/projects/{name}/activate")
def activate_project(name: str):
    registered = workspace_manager.load_and_register_project_cards(name)
    schemas = list_cards()
    return {"registered": registered, "cards": schemas}
