from __future__ import annotations

import io
import base64
import torch

from cards.base import BaseCard


class TrainingBackwardCard(BaseCard):
    card_type = "training_backward"
    display_name = "Backward Pass"
    description = "Compute gradients via backpropagation"
    category = "training"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "type": "object",
        "properties": {
            "retain_graph": {
                "type": "boolean",
                "description": "Retain computation graph for multiple backward passes",
                "default": False,
            },
        },
    }
    input_schema = {
        "loss": "json",
        "model": "model",
    }
    output_schema = {
        "model": "model",  # Model with computed gradients
        "gradients_computed": "json",
    }

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load loss
        loss_data = storage.load_json(inputs["loss"])
        loss_bytes = base64.b64decode(loss_data["data"])
        loss = torch.load(io.BytesIO(loss_bytes), map_location=device)

        # Load model
        model = storage.load_model(inputs["model"])
        model = model.to(device)
        model.train()

        # Backward pass
        retain_graph = config.get("retain_graph", False)
        loss.backward(retain_graph=retain_graph)

        # Check if gradients were computed
        has_gradients = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )

        # Save model with gradients
        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        model_ref = storage.save_model(pid, nid, "model", model.cpu())

        gradients_ref = storage.save_json(pid, nid, "gradients_computed", {
            "has_gradients": has_gradients,
            "retain_graph": retain_graph,
        })

        return {
            "model": model_ref,
            "gradients_computed": gradients_ref,
        }

    def get_output_preview(self, outputs: dict, storage) -> dict:
        grad_data = storage.load_json(outputs["gradients_computed"])
        return {
            "gradients_computed": grad_data["has_gradients"],
            "retain_graph": grad_data["retain_graph"],
        }
