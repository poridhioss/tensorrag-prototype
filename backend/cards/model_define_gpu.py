from __future__ import annotations

from cards.base import BaseCard

VALID_GPU_MODEL_TYPES = {"neural_network", "deep_neural_network"}


class ModelDefineGPUCard(BaseCard):
    card_type = "model_define_gpu"
    display_name = "Model Define (GPU)"
    description = "Define neural network architecture for GPU training"
    category = "model"
    execution_mode = "local"  # Config-only, no execution
    output_view_type = "model_summary"

    config_schema = {
        "type": "object",
        "properties": {
            "model_type": {
                "type": "string",
                "enum": ["neural_network", "deep_neural_network"],
                "description": "Type of neural network",
                "default": "neural_network",
            },
            "hidden_layers": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Number of neurons in each hidden layer",
                "default": [64, 32],
            },
            "activation": {
                "type": "string",
                "enum": ["relu", "tanh", "sigmoid"],
                "description": "Activation function",
                "default": "relu",
            },
            "dropout": {
                "type": "number",
                "description": "Dropout rate (0-1)",
                "default": 0.2,
                "minimum": 0,
                "maximum": 1,
            },
            "learning_rate": {
                "type": "number",
                "description": "Learning rate",
                "default": 0.001,
                "minimum": 0.0001,
                "maximum": 0.1,
            },
            "epochs": {
                "type": "integer",
                "description": "Number of training epochs",
                "default": 50,
                "minimum": 1,
                "maximum": 1000,
            },
            "batch_size": {
                "type": "integer",
                "description": "Batch size",
                "default": 32,
                "minimum": 1,
                "maximum": 1024,
            },
        },
        "required": ["model_type"],
    }
    input_schema = {}
    output_schema = {"model_spec": "json"}

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        model_type = config["model_type"]
        if model_type not in VALID_GPU_MODEL_TYPES:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Valid types: {VALID_GPU_MODEL_TYPES}"
            )

        spec = {
            "model_type": model_type,
            "hidden_layers": config.get("hidden_layers", [64, 32]),
            "activation": config.get("activation", "relu"),
            "dropout": config.get("dropout", 0.2),
            "learning_rate": config.get("learning_rate", 0.001),
            "epochs": config.get("epochs", 50),
            "batch_size": config.get("batch_size", 32),
        }

        ref = storage.save_json(
            config["_pipeline_id"], config["_node_id"], "model_spec", spec
        )
        return {"model_spec": ref}

    def get_output_preview(self, outputs: dict, storage) -> dict:
        spec = storage.load_json(outputs["model_spec"])
        return {
            "model_type": spec["model_type"],
            "architecture": {
                "hidden_layers": spec["hidden_layers"],
                "activation": spec["activation"],
                "dropout": spec["dropout"],
            },
            "training": {
                "learning_rate": spec["learning_rate"],
                "epochs": spec["epochs"],
                "batch_size": spec["batch_size"],
            },
        }
