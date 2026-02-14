from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

from cards.base import BaseCard

MODEL_CLASSES = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
}


class TrainCard(BaseCard):
    card_type = "train"
    display_name = "Train"
    description = "Train a model on the training dataset"
    category = "model"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "type": "object",
        "properties": {
            "target_column": {
                "type": "string",
                "description": "Name of the target column",
            },
            "feature_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Feature columns (empty = all except target)",
            },
        },
        "required": ["target_column"],
    }
    input_schema = {"train_dataset": "dataframe", "model_spec": "json"}
    output_schema = {"trained_model": "model", "train_metrics": "json"}

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        train_df = storage.load_dataframe(inputs["train_dataset"])
        model_spec = storage.load_json(inputs["model_spec"])

        target_col = config["target_column"]
        feature_cols = config.get("feature_columns", [])
        if not feature_cols:
            feature_cols = [c for c in train_df.columns if c != target_col]

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values

        model_type = model_spec["model_type"]
        hyperparams = model_spec.get("hyperparameters", {})

        model_cls = MODEL_CLASSES.get(model_type)
        if model_cls is None:
            raise ValueError(f"Unknown model type: {model_type}")

        model = model_cls(**hyperparams)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        metrics = {
            "train_mse": float(mean_squared_error(y_train, y_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred))),
            "train_r2": float(r2_score(y_train, y_pred)),
            "feature_columns": feature_cols,
            "target_column": target_col,
        }

        if hasattr(model, "coef_"):
            metrics["coefficients"] = {
                col: float(coef)
                for col, coef in zip(feature_cols, model.coef_)
            }
        if hasattr(model, "intercept_"):
            metrics["intercept"] = float(model.intercept_)

        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        model_ref = storage.save_model(pid, nid, "trained_model", model)
        # Save model metadata so inference card knows feature columns
        storage.save_json(pid, nid, "model_meta", {
            "feature_columns": feature_cols,
            "target_column": target_col,
        })
        metrics_ref = storage.save_json(pid, nid, "train_metrics", metrics)

        return {"trained_model": model_ref, "train_metrics": metrics_ref}

    def get_output_preview(self, outputs: dict, storage) -> dict:
        metrics = storage.load_json(outputs["train_metrics"])
        return {
            "metrics": {
                "train_mse": metrics["train_mse"],
                "train_rmse": metrics["train_rmse"],
                "train_r2": metrics["train_r2"],
            },
            "coefficients": metrics.get("coefficients", {}),
            "intercept": metrics.get("intercept"),
            "feature_columns": metrics["feature_columns"],
            "target_column": metrics["target_column"],
        }
