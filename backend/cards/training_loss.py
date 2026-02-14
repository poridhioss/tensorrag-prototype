from __future__ import annotations

import io
import base64
import torch
import torch.nn as nn

from cards.base import BaseCard


class TrainingLossCard(BaseCard):
    card_type = "training_loss"
    display_name = "Calculate Loss"
    description = "Calculate loss between predictions and targets"
    category = "training"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {
        "type": "object",
        "properties": {
            "loss_type": {
                "type": "string",
                "enum": ["mse", "mae", "huber"],
                "description": "Type of loss function",
                "default": "mse",
            },
        },
    }
    input_schema = {
        "predictions": "json",
        "batch_data": "json",  # Contains targets in batch_data["y"]
    }
    output_schema = {
        "loss": "json",
        "loss_value": "json",
    }

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Validate inputs
        if "predictions" not in inputs:
            raise ValueError(
                "Calculate Loss requires 'predictions' input. "
                f"Received inputs: {list(inputs.keys())}"
            )
        if "batch_data" not in inputs:
            raise ValueError(
                "Calculate Loss requires 'batch_data' input. "
                f"Received inputs: {list(inputs.keys())}. "
                "Make sure 'Prepare Batch' is connected to 'Calculate Loss' with 'batch_data' output."
            )

        # Load predictions and batch data (contains targets)
        try:
            pred_data = storage.load_json(inputs["predictions"])
        except Exception as e:
            raise ValueError(
                f"Failed to load predictions from storage: {e}. "
                f"Input ref: {inputs.get('predictions')}"
            )
        
        try:
            batch_data = storage.load_json(inputs["batch_data"])
        except Exception as e:
            raise ValueError(
                f"Failed to load batch_data from storage: {e}. "
                f"Input ref: {inputs.get('batch_data')}"
            )

        # Validate data structure
        if not isinstance(pred_data, dict):
            raise ValueError(
                f"Predictions data is not a dict. Type: {type(pred_data)}, Value: {pred_data}"
            )
        if not isinstance(batch_data, dict):
            raise ValueError(
                f"Batch data is not a dict. Type: {type(batch_data)}, Value: {batch_data}"
            )
        
        if "data" not in pred_data:
            raise ValueError(
                f"Predictions data missing 'data' key. Available keys: {list(pred_data.keys())}"
            )
        if "y" not in batch_data:
            raise ValueError(
                f"Batch data missing 'y' key. Available keys: {list(batch_data.keys())}"
            )

        # Deserialize tensors
        try:
            pred_bytes = base64.b64decode(pred_data["data"])
            target_bytes = base64.b64decode(batch_data["y"])
        except KeyError as e:
            raise ValueError(
                f"KeyError accessing data: {e}. "
                f"pred_data keys: {list(pred_data.keys()) if isinstance(pred_data, dict) else 'N/A'}, "
                f"batch_data keys: {list(batch_data.keys()) if isinstance(batch_data, dict) else 'N/A'}"
            )
        predictions = torch.load(io.BytesIO(pred_bytes), map_location=device)
        targets = torch.load(io.BytesIO(target_bytes), map_location=device)

        # Select loss function
        loss_type = config.get("loss_type", "mse")
        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mae":
            criterion = nn.L1Loss()
        elif loss_type == "huber":
            criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Calculate loss
        loss = criterion(predictions, targets)
        loss_value = float(loss.item())

        # Serialize loss tensor
        loss_buf = io.BytesIO()
        torch.save(loss, loss_buf)
        loss_bytes = base64.b64encode(loss_buf.getvalue()).decode()

        # Save loss
        pid = config["_pipeline_id"]
        nid = config["_node_id"]

        loss_ref = storage.save_json(pid, nid, "loss", {
            "data": loss_bytes,
            "value": loss_value,
            "loss_type": loss_type,
            "requires_grad": loss.requires_grad,
        })

        loss_value_ref = storage.save_json(pid, nid, "loss_value", {
            "value": loss_value,
            "loss_type": loss_type,
        })

        return {
            "loss": loss_ref,
            "loss_value": loss_value_ref,
        }

    def get_output_preview(self, outputs: dict, storage) -> dict:
        loss_data = storage.load_json(outputs["loss_value"])
        return {
            "loss_value": loss_data["value"],
            "loss_type": loss_data["loss_type"],
        }
