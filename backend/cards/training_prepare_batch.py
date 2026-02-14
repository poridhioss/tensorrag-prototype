from __future__ import annotations

import io
import base64
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cards.base import BaseCard


class TrainingPrepareBatchCard(BaseCard):
    card_type = "training_prepare_batch"
    display_name = "Prepare Batch"
    description = "Prepare a batch of data for training"
    category = "training"
    execution_mode = "modal"
    output_view_type = "table"

    config_schema = {
        "type": "object",
        "properties": {
            "batch_index": {
                "type": "integer",
                "description": "Batch index to retrieve",
                "default": 0,
            },
            "batch_size": {
                "type": "integer",
                "description": "Batch size",
                "default": 32,
            },
        },
    }
    input_schema = {
        "train_dataset": "dataframe",
    }
    output_schema = {
        "batch_data": "json",  # Serialized batch tensors
        "batch_info": "json",
    }

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        import pandas as pd

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        train_df = storage.load_dataframe(inputs["train_dataset"])

        # Convert to tensors
        X = torch.FloatTensor(train_df.values).to(device)
        # For now, assume last column is target (will be improved)
        # In practice, you'd split features and target separately
        y = X[:, -1:].clone()
        X = X[:, :-1]

        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        batch_size = config.get("batch_size", 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Get specific batch
        batch_index = config.get("batch_index", 0)
        batch_X, batch_y = list(dataloader)[batch_index]

        # Serialize tensors
        X_buf = io.BytesIO()
        y_buf = io.BytesIO()
        torch.save(batch_X, X_buf)
        torch.save(batch_y, y_buf)
        X_bytes = base64.b64encode(X_buf.getvalue()).decode()
        y_bytes = base64.b64encode(y_buf.getvalue()).decode()

        # Save batch data
        pid = config["_pipeline_id"]
        nid = config["_node_id"]

        batch_data_ref = storage.save_json(pid, nid, "batch_data", {
            "X": X_bytes,
            "y": y_bytes,
            "batch_size": len(batch_X),
        })

        batch_info_ref = storage.save_json(pid, nid, "batch_info", {
            "batch_index": batch_index,
            "batch_size": len(batch_X),
            "X_shape": list(batch_X.shape),
            "y_shape": list(batch_y.shape),
        })

        return {
            "batch_data": batch_data_ref,
            "batch_info": batch_info_ref,
        }

    def get_output_preview(self, outputs: dict, storage) -> dict:
        batch_info = storage.load_json(outputs["batch_info"])
        return {
            "batch_index": batch_info["batch_index"],
            "batch_size": batch_info["batch_size"],
            "X_shape": batch_info["X_shape"],
            "y_shape": batch_info["y_shape"],
        }
