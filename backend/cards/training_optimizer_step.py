from __future__ import annotations

import torch
import torch.optim as optim

from cards.base import BaseCard


class TrainingOptimizerStepCard(BaseCard):
    card_type = "training_optimizer_step"
    display_name = "Optimizer Step"
    description = "Update model parameters using optimizer"
    category = "training"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "type": "object",
        "properties": {
            "gradient_clip": {
                "type": "number",
                "description": "Gradient clipping value (0 = no clipping)",
                "default": 0,
                "minimum": 0,
            },
        },
    }
    input_schema = {
        "optimizer": "model",
        "model": "model",
    }
    output_schema = {
        "model": "model",  # Updated model
        "optimizer": "model",  # Updated optimizer
        "step_info": "json",
    }

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and optimizer
        model = storage.load_model(inputs["model"])
        optimizer = storage.load_model(inputs["optimizer"])

        model = model.to(device)
        
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # Gradient clipping if specified
        gradient_clip = config.get("gradient_clip", 0)
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Optimizer step
        optimizer.step()

        # Get step info
        step_info = {
            "gradient_clip": gradient_clip,
            "device": str(device),
        }

        # Save updated model and optimizer
        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        model_ref = storage.save_model(pid, nid, "model", model.cpu())
        optimizer_ref = storage.save_model(pid, nid, "optimizer", optimizer)

        step_info_ref = storage.save_json(pid, nid, "step_info", step_info)

        return {
            "model": model_ref,
            "optimizer": optimizer_ref,
            "step_info": step_info_ref,
        }

    def get_output_preview(self, outputs: dict, storage) -> dict:
        step_info = storage.load_json(outputs["step_info"])
        return {
            "gradient_clip": step_info["gradient_clip"],
            "device": step_info["device"],
            "status": "Parameters updated",
        }
