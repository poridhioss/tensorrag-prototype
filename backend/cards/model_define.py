from __future__ import annotations

from cards.base import BaseCard

VALID_MODEL_TYPES = {"linear_regression", "ridge", "lasso"}


class ModelDefineCard(BaseCard):
    card_type = "model_define"
    display_name = "Model Define"
    description = "Define model type and hyperparameters"
    category = "model"
    execution_mode = "local"
    output_view_type = "model_summary"

    config_schema = {
        "type": "object",
        "properties": {
            "model_type": {
                "type": "string",
                "enum": ["linear_regression", "ridge", "lasso"],
                "description": "Type of model to train",
            },
            "hyperparameters": {
                "type": "object",
                "description": "Model hyperparameters (e.g., alpha for ridge/lasso)",
                "default": {},
            },
        },
        "required": ["model_type"],
    }
    input_schema = {}
    output_schema = {"model_spec": "json"}

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        model_type = config["model_type"]
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Valid types: {VALID_MODEL_TYPES}"
            )

        spec = {
            "model_type": model_type,
            "hyperparameters": config.get("hyperparameters", {}),
        }

        ref = storage.save_json(
            config["_pipeline_id"], config["_node_id"], "model_spec", spec
        )
        return {"model_spec": ref}

    def get_output_preview(self, outputs: dict, storage) -> dict:
        spec = storage.load_json(outputs["model_spec"])
        return {
            "model_type": spec["model_type"],
            "hyperparameters": spec["hyperparameters"],
        }
