from __future__ import annotations

import io
import base64
import torch

from cards.base import BaseCard


class TrainingForwardCard(BaseCard):
    card_type = "training_forward"
    display_name = "Forward Pass"
    description = "Perform forward pass through the model"
    category = "training"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "type": "object",
        "properties": {
            "batch_index": {
                "type": "integer",
                "description": "Current batch index (for tracking)",
                "default": 0,
            },
        },
    }
    input_schema = {
        "model": "model",
        "batch_data": "json",  # Contains X and y tensors as serialized
    }
    output_schema = {
        "predictions": "json",  # Serialized tensor
        "model": "model",  # Updated model state
    }

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        import joblib
        import json
        import base64

        # Load model
        loaded_obj = storage.load_model(inputs["model"])
        
        # Validate that we got a model, not an optimizer
        if not hasattr(loaded_obj, "forward") and not hasattr(loaded_obj, "__call__"):
            # Check if it's an optimizer
            if hasattr(loaded_obj, "step") and hasattr(loaded_obj, "zero_grad"):
                raise ValueError(
                    "Forward Pass received an optimizer instead of a model. "
                    "Connect the 'model' output from Initialize Optimizer, not the 'optimizer' output."
                )
            raise ValueError(f"Forward Pass received invalid object type: {type(loaded_obj)}")
        
        model = loaded_obj
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        # Load batch data
        batch_data = storage.load_json(inputs["batch_data"])
        
        # Deserialize tensors
        X_bytes = base64.b64decode(batch_data["X"])
        y_bytes = base64.b64decode(batch_data["y"])
        X = torch.load(io.BytesIO(X_bytes), map_location=device)
        y = torch.load(io.BytesIO(y_bytes), map_location=device)

        # Forward pass
        with torch.set_grad_enabled(True):
            predictions = model(X)

        # Serialize predictions
        pred_buf = io.BytesIO()
        torch.save(predictions, pred_buf)
        pred_bytes = base64.b64encode(pred_buf.getvalue()).decode()

        # Save predictions and model
        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        
        predictions_ref = storage.save_json(pid, nid, "predictions", {
            "data": pred_bytes,
            "shape": list(predictions.shape),
            "device": str(device),
            "batch_index": config.get("batch_index", 0),
        })
        
        model_ref = storage.save_model(pid, nid, "model", model.cpu())

        return {
            "predictions": predictions_ref,
            "model": model_ref,
        }

    def get_output_preview(self, outputs: dict, storage) -> dict:
        pred_data = storage.load_json(outputs["predictions"])
        return {
            "predictions_shape": pred_data["shape"],
            "device": pred_data["device"],
            "batch_index": pred_data["batch_index"],
        }
