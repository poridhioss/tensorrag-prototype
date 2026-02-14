from __future__ import annotations

import io
import base64
import torch
import torch.optim as optim

from cards.base import BaseCard


class TrainingZeroGradCard(BaseCard):
    card_type = "training_zero_grad"
    display_name = "Zero Gradients"
    description = "Zero out gradients in the optimizer"
    category = "training"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "type": "object",
        "properties": {},
    }
    input_schema = {
        "optimizer": "model",  # Optimizer state saved as model
    }
    output_schema = {
        "optimizer": "model",  # Optimizer with zeroed gradients
    }

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        # Load optimizer
        optimizer = storage.load_model(inputs["optimizer"])

        # Zero gradients
        optimizer.zero_grad()

        # Save optimizer
        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        optimizer_ref = storage.save_model(pid, nid, "optimizer", optimizer)

        return {"optimizer": optimizer_ref}

    def get_output_preview(self, outputs: dict, storage) -> dict:
        return {
            "status": "Gradients zeroed",
        }
