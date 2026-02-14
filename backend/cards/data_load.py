from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_california_housing

from cards.base import BaseCard


class DataLoadCard(BaseCard):
    card_type = "data_load"
    display_name = "Data Load"
    description = "Load data from CSV, URL, or a sample dataset"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["csv", "url", "sample"],
                "description": "Data source type",
            },
            "path": {"type": "string", "description": "Local file path (for csv)"},
            "url": {"type": "string", "description": "Remote URL (for url)"},
            "sample_name": {
                "type": "string",
                "enum": ["california_housing", "boston_housing"],
                "description": "Sample dataset name",
            },
        },
        "required": ["source"],
    }
    input_schema = {}
    output_schema = {"dataset": "dataframe"}

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        source = config["source"]

        if source == "csv":
            df = pd.read_csv(config["path"])
        elif source == "url":
            df = pd.read_csv(config["url"])
        elif source == "sample":
            sample_name = config.get("sample_name", "california_housing")
            if sample_name == "california_housing":
                housing = fetch_california_housing(as_frame=True)
                df = housing.frame
            elif sample_name == "boston_housing":
                # Create a larger synthetic dataset based on Boston housing patterns
                # This is suitable for GPU training
                from sklearn.datasets import make_regression
                import numpy as np
                X, y = make_regression(
                    n_samples=50000,  # Larger dataset for GPU training
                    n_features=13,
                    n_informative=10,
                    noise=10.0,
                    random_state=42
                )
                feature_names = [
                    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
                ]
                df = pd.DataFrame(X, columns=feature_names)
                df["MEDV"] = y  # Median value (target)
            else:
                raise ValueError(f"Unknown sample dataset: {sample_name}")
        else:
            raise ValueError(f"Unknown source: {source}")

        ref = storage.save_dataframe(
            config["_pipeline_id"], config["_node_id"], "dataset", df
        )
        return {"dataset": ref}

    def get_output_preview(self, outputs: dict, storage) -> dict:
        df = storage.load_dataframe(outputs["dataset"])
        return {
            "rows": df.head(100).to_dict(orient="records"),
            "columns": [
                {"name": col, "dtype": str(df[col].dtype)} for col in df.columns
            ],
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "null_counts": df.isnull().sum().to_dict(),
        }
