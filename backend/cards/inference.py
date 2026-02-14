from __future__ import annotations

import pandas as pd

from cards.base import BaseCard


class InferenceCard(BaseCard):
    card_type = "inference"
    display_name = "Inference"
    description = "Run predictions on new data using a trained model"
    category = "inference"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "type": "object",
        "properties": {
            "output_format": {
                "type": "string",
                "enum": ["parquet", "csv"],
                "default": "parquet",
                "description": "Output file format",
            },
            "include_features": {
                "type": "boolean",
                "default": True,
                "description": "Include original features alongside predictions",
            },
            "target_column": {
                "type": "string",
                "description": "Name for the prediction column",
                "default": "prediction",
            },
            "feature_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Feature columns to use (empty = all)",
            },
        },
    }
    input_schema = {"trained_model": "model", "dataset": "dataframe"}
    output_schema = {"predictions": "dataframe"}

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        model = storage.load_model(inputs["trained_model"])

        # Accept either "dataset" or "test_dataset" as input
        dataset_ref = inputs.get("dataset") or inputs.get("test_dataset")
        if dataset_ref is None:
            raise ValueError("Inference card requires a dataset input")
        data_df = storage.load_dataframe(dataset_ref)

        feature_cols = config.get("feature_columns", [])
        target_col = config.get("target_column", "prediction")

        if not feature_cols:
            # Try to read feature columns from model metadata saved by train card
            model_ref = inputs["trained_model"]
            meta_path = str(model_ref).replace("trained_model.joblib", "model_meta.json")
            try:
                meta = storage.load_json(meta_path)
                feature_cols = meta.get("feature_columns", [])
            except Exception:
                pass

        if not feature_cols:
            # Fallback: use all columns, excluding the prediction target column
            feature_cols = [c for c in data_df.columns if c != target_col]

        X = data_df[feature_cols].values
        predictions = model.predict(X)

        if config.get("include_features", True):
            result_df = data_df[feature_cols].copy()
            result_df[target_col] = predictions
        else:
            result_df = pd.DataFrame({target_col: predictions})

        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        ref = storage.save_dataframe(pid, nid, "predictions", result_df)

        return {"predictions": ref}

    def get_output_preview(self, outputs: dict, storage) -> dict:
        df = storage.load_dataframe(outputs["predictions"])
        return {
            "rows": df.head(100).to_dict(orient="records"),
            "columns": [
                {"name": col, "dtype": str(df[col].dtype)} for col in df.columns
            ],
            "row_count": len(df),
        }
