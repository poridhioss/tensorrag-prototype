from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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


class TrainGPUCard(BaseCard):
    card_type = "train_gpu"
    display_name = "Train (GPU)"
    description = "Train a neural network on GPU using PyTorch"
    category = "model"
    execution_mode = "modal"  # Runs on Modal with GPU
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
        import pandas as pd

        train_df = storage.load_dataframe(inputs["train_dataset"])
        model_spec = storage.load_json(inputs["model_spec"])

        target_col = config["target_column"]
        feature_cols = config.get("feature_columns", [])
        if not feature_cols:
            feature_cols = [c for c in train_df.columns if c != target_col]

        # Prepare data
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[target_col].values.astype(np.float32).reshape(-1, 1)

        # Normalize features
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - X_mean) / X_std

        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8
        y_train = (y_train - y_mean) / y_std

        # Convert to PyTorch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).to(device)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = model_spec.get("batch_size", 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Build model
        input_size = len(feature_cols)
        hidden_layers = model_spec.get("hidden_layers", [64, 32])
        activation = model_spec.get("activation", "relu")
        dropout = model_spec.get("dropout", 0.2)

        model = NeuralNetwork(input_size, hidden_layers, activation, dropout).to(device)
        criterion = nn.MSELoss()
        learning_rate = model_spec.get("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        epochs = model_spec.get("epochs", 50)
        train_losses = []

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)

        # Evaluate on training set
        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_tensor)
            y_pred = y_pred_tensor.cpu().numpy()
            y_train_denorm = y_train * y_std + y_mean
            y_pred_denorm = y_pred * y_std + y_mean

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score

        mse = float(mean_squared_error(y_train_denorm, y_pred_denorm))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_train_denorm, y_pred_denorm))

        metrics = {
            "train_mse": mse,
            "train_rmse": rmse,
            "train_r2": r2,
            "final_loss": train_losses[-1],
            "loss_history": train_losses,
            "feature_columns": feature_cols,
            "target_column": target_col,
            "device_used": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs_trained": epochs,
        }

        # Save model (move to CPU for serialization)
        model_cpu = model.cpu()
        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        model_ref = storage.save_model(pid, nid, "trained_model", model_cpu)

        # Save normalization parameters
        storage.save_json(pid, nid, "model_meta", {
            "feature_columns": feature_cols,
            "target_column": target_col,
            "X_mean": X_mean.tolist(),
            "X_std": X_std.tolist(),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
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
                "final_loss": metrics["final_loss"],
            },
            "device_used": metrics["device_used"],
            "epochs_trained": metrics["epochs_trained"],
            "loss_history": metrics["loss_history"][-10:],  # Last 10 epochs
            "feature_columns": metrics["feature_columns"],
            "target_column": metrics["target_column"],
        }
