from __future__ import annotations

import torch
import torch.nn as nn

from cards.base import BaseCard


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_layers: list[int], activation: str, dropout: float):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TrainingBuildModelCard(BaseCard):
    card_type = "training_build_model"
    display_name = "Build Model"
    description = "Build neural network model from specification"
    category = "training"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "type": "object",
        "properties": {
            "input_size": {
                "type": "integer",
                "description": "Number of input features",
                "default": 13,
                "minimum": 1,
            },
        },
        "required": ["input_size"],
    }
    input_schema = {
        "model_spec": "json",
    }
    output_schema = {
        "model": "model",
    }

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model specification
        model_spec = storage.load_json(inputs["model_spec"])

        # Get architecture parameters
        input_size = config["input_size"]
        hidden_layers = model_spec.get("hidden_layers", [64, 32])
        activation = model_spec.get("activation", "relu")
        dropout = model_spec.get("dropout", 0.2)

        # Build model
        model = NeuralNetwork(input_size, hidden_layers, activation, dropout).to(device)

        # Save model
        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        model_ref = storage.save_model(pid, nid, "model", model.cpu())

        return {"model": model_ref}

    def get_output_preview(self, outputs: dict, storage) -> dict:
        return {
            "status": "Model built",
        }
