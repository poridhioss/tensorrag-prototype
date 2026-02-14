from __future__ import annotations

import torch
import torch.optim as optim

from cards.base import BaseCard


class TrainingInitOptimizerCard(BaseCard):
    card_type = "training_init_optimizer"
    display_name = "Initialize Optimizer"
    description = "Initialize optimizer for model training"
    category = "training"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "type": "object",
        "properties": {
            "optimizer_type": {
                "type": "string",
                "enum": ["adam", "sgd", "rmsprop"],
                "description": "Type of optimizer",
                "default": "adam",
            },
            "learning_rate": {
                "type": "number",
                "description": "Learning rate",
                "default": 0.001,
                "minimum": 0.0001,
                "maximum": 0.1,
            },
            "weight_decay": {
                "type": "number",
                "description": "Weight decay (L2 regularization)",
                "default": 0.0,
                "minimum": 0,
            },
        },
        "required": ["optimizer_type", "learning_rate"],
    }
    input_schema = {
        "model": "model",
    }
    output_schema = {
        "optimizer": "model",  # Optimizer saved as model
        "model": "model",  # Model passes through
    }

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        model = storage.load_model(inputs["model"])
        model = model.to(device)

        # Initialize optimizer
        optimizer_type = config["optimizer_type"]
        learning_rate = config["learning_rate"]
        weight_decay = config.get("weight_decay", 0.0)

        if optimizer_type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Save optimizer and model (model passes through)
        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        optimizer_ref = storage.save_model(pid, nid, "optimizer", optimizer)
        model_ref = storage.save_model(pid, nid, "model", model.cpu())

        return {"optimizer": optimizer_ref, "model": model_ref}

    def get_output_preview(self, outputs: dict, storage) -> dict:
        return {
            "status": "Optimizer initialized",
        }
