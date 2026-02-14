from __future__ import annotations

import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cards.base import BaseCard

matplotlib.use("Agg")

METRIC_FUNCTIONS = {
    "mse": ("MSE", mean_squared_error),
    "rmse": ("RMSE", lambda y, p: float(np.sqrt(mean_squared_error(y, p)))),
    "mae": ("MAE", mean_absolute_error),
    "r2": ("R2", r2_score),
}


class EvaluateCard(BaseCard):
    card_type = "evaluate"
    display_name = "Evaluate"
    description = "Evaluate model performance on the test set"
    category = "evaluation"
    execution_mode = "local"
    output_view_type = "metrics"

    config_schema = {
        "type": "object",
        "properties": {
            "metrics": {
                "type": "array",
                "items": {"type": "string", "enum": ["mse", "rmse", "mae", "r2"]},
                "default": ["mse", "r2"],
                "description": "Metrics to compute",
            },
            "target_column": {
                "type": "string",
                "description": "Target column name",
            },
            "feature_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Feature columns (empty = all except target)",
            },
        },
        "required": ["target_column"],
    }
    input_schema = {"trained_model": "model", "test_dataset": "dataframe"}
    output_schema = {"eval_report": "json", "eval_chart": "bytes"}

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        model = storage.load_model(inputs["trained_model"])
        test_df = storage.load_dataframe(inputs["test_dataset"])

        target_col = config["target_column"]
        feature_cols = config.get("feature_columns", [])
        if not feature_cols:
            feature_cols = [c for c in test_df.columns if c != target_col]

        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        y_pred = model.predict(X_test)

        requested_metrics = config.get("metrics", ["mse", "r2"])
        results = {}
        for m in requested_metrics:
            if m in METRIC_FUNCTIONS:
                name, fn = METRIC_FUNCTIONS[m]
                results[name] = float(fn(y_test, y_pred))

        # Generate predicted-vs-actual scatter plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, s=10)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            lw=1,
        )
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        chart_bytes = buf.getvalue()

        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        report_ref = storage.save_json(pid, nid, "eval_report", results)
        chart_ref = storage.save_bytes(pid, nid, "eval_chart", chart_bytes, "png")

        return {"eval_report": report_ref, "eval_chart": chart_ref}

    def get_output_preview(self, outputs: dict, storage) -> dict:
        report = storage.load_json(outputs["eval_report"])
        return {
            "metrics": report,
            "chart_ref": outputs.get("eval_chart"),
        }
