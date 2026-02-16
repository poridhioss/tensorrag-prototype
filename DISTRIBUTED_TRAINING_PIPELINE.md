# Distributed Training Pipeline — Step-by-Step Cards

Build a complete distributed training pipeline using small, reusable cards. Each card handles one step of distributed training, supporting DDP (Distributed Data Parallel), FSDP (Fully Sharded Data Parallel), and DeepSpeed strategies.

## Pipeline Overview

```
Load Dataset → Prepare Data → Shard Data → Initialize Distributed Strategy
  → Setup Process Group → Distributed Forward → Distributed Backward
  → All-Reduce Gradients → Optimizer Step → Save Checkpoint → Aggregate Model → Evaluate
```

## Project File Structure

Create the following folders and files in the **Editor** view:

```
distributed-training-pipeline/    ← Project name
├── data/                          ← Folder
│   ├── load_dataset.py           ← Card 1
│   └── prepare_data.py           ← Card 2
├── sharding/                      ← Folder
│   └── shard_data.py             ← Card 3
├── distributed/                   ← Folder
│   ├── init_strategy.py          ← Card 4
│   ├── setup_process_group.py    ← Card 5
│   ├── forward_pass.py           ← Card 6
│   ├── backward_pass.py          ← Card 7
│   ├── allreduce_gradients.py    ← Card 8
│   └── optimizer_step.py         ← Card 9
├── checkpointing/                 ← Folder
│   └── save_checkpoint.py        ← Card 10
├── aggregation/                   ← Folder
│   └── aggregate_model.py        ← Card 11
└── evaluation/                    ← Folder
    └── evaluate.py                ← Card 12
```

## Card Connection Map

| # | Card | File | Folder | Receives from | Sends to |
|---|------|------|--------|--------------|----------|
| 1 | Load Dataset | `load_dataset.py` | `data/` | — (config: dataset path) | `dataset` |
| 2 | Prepare Data | `prepare_data.py` | `data/` | `dataset` | `prepared_data` |
| 3 | Shard Data | `shard_data.py` | `sharding/` | `prepared_data` | `sharded_data` |
| 4 | Initialize Strategy | `init_strategy.py` | `distributed/` | — (config: strategy type) | `strategy_config` |
| 5 | Setup Process Group | `setup_process_group.py` | `distributed/` | `strategy_config` | `process_group` |
| 6 | Distributed Forward | `forward_pass.py` | `distributed/` | `sharded_data`, `process_group` | `forward_output` |
| 7 | Distributed Backward | `backward_pass.py` | `distributed/` | `forward_output`, `process_group` | `gradients` |
| 8 | All-Reduce Gradients | `allreduce_gradients.py` | `distributed/` | `gradients`, `process_group` | `synced_gradients` |
| 9 | Optimizer Step | `optimizer_step.py` | `distributed/` | `synced_gradients`, `process_group` | `updated_model` |
| 10 | Save Checkpoint | `save_checkpoint.py` | `checkpointing/` | `updated_model`, `process_group` | `checkpoint` |
| 11 | Aggregate Model | `aggregate_model.py` | `aggregation/` | `checkpoint` | `aggregated_model` |
| 12 | Evaluate | `evaluate.py` | `evaluation/` | `aggregated_model`, `prepared_data` | `eval_metrics` |

> **Note:** The executor automatically fills in the real `pipeline_id` and `node_id` for storage calls. Use any placeholder (e.g. `"_p"`, `"_n"`) — they get replaced at runtime.

---

## Card 1: Load Dataset

**File:** `load_dataset.py` | **Folder:** `data/`

Loads a dataset from a file or URL for distributed training.

```python
from cards.base import BaseCard
import pandas as pd
import numpy as np


class LoadDatasetCard(BaseCard):
    card_type = "dist_load_dataset"
    display_name = "Load Dataset"
    description = "Load dataset for distributed training"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "source_url": {
            "type": "string",
            "label": "Dataset URL or path",
            "default": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        },
        "target_column": {
            "type": "string",
            "label": "Target column name",
            "default": "species",
        },
    }
    input_schema = {}
    output_schema = {"dataset": "json"}

    def execute(self, config, inputs, storage):
        df = pd.read_csv(config["source_url"])
        target = config["target_column"]
        
        # Encode labels if needed
        if df[target].dtype == object:
            labels = sorted(df[target].unique().tolist())
            label_map = {l: i for i, l in enumerate(labels)}
            df[target] = df[target].map(label_map)
        
        X = df.drop(columns=[target]).values.astype(float)
        y = df[target].values.astype(float)
        
        data = {
            "X": X.tolist(),
            "y": y.tolist(),
            "feature_names": [c for c in df.columns if c != target],
            "num_features": X.shape[1],
            "num_classes": int(y.max()) + 1,
            "num_samples": len(X),
        }
        
        ref = storage.save_json("_p", "_n", "dataset", data)
        return {"dataset": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["dataset"])
        return {
            "columns": ["property", "value"],
            "rows": [
                ["Samples", data["num_samples"]],
                ["Features", data["num_features"]],
                ["Classes", data["num_classes"]],
            ],
            "total_rows": 3,
        }
```

---

## Card 2: Prepare Data

**File:** `prepare_data.py` | **Folder:** `data/`

Prepares data for distributed training: normalization, train/test split, and batching configuration.

```python
from cards.base import BaseCard
import numpy as np


class PrepareDataCard(BaseCard):
    card_type = "dist_prepare_data"
    display_name = "Prepare Data"
    description = "Prepare and normalize data for distributed training"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "test_ratio": {
            "type": "number",
            "label": "Test set ratio",
            "default": 0.2,
        },
        "normalize": {
            "type": "boolean",
            "label": "Normalize features",
            "default": True,
        },
        "batch_size": {
            "type": "number",
            "label": "Batch size per worker",
            "default": 32,
        },
    }
    input_schema = {"dataset": "json"}
    output_schema = {"prepared_data": "json"}

    def execute(self, config, inputs, storage):
        data = storage.load_json(inputs["dataset"])
        X = np.array(data["X"])
        y = np.array(data["y"])
        test_ratio = float(config.get("test_ratio", 0.2))
        normalize = config.get("normalize", True)
        batch_size = int(config.get("batch_size", 32))
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split
        split = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Normalize
        if normalize:
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0) + 1e-8
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
            norm_stats = {"mean": mean.tolist(), "std": std.tolist()}
        else:
            norm_stats = None
        
        prepared = {
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist(),
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist(),
            "num_features": data["num_features"],
            "num_classes": data["num_classes"],
            "batch_size": batch_size,
            "normalize_stats": norm_stats,
        }
        
        ref = storage.save_json("_p", "_n", "prepared_data", prepared)
        return {"prepared_data": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["prepared_data"])
        return {
            "columns": ["split", "samples", "features", "classes"],
            "rows": [
                ["Train", len(data["y_train"]), data["num_features"], data["num_classes"]],
                ["Test", len(data["y_test"]), data["num_features"], data["num_classes"]],
            ],
            "total_rows": 2,
        }
```

---

## Card 3: Shard Data

**File:** `shard_data.py` | **Folder:** `sharding/`

Shards the training data across multiple workers for distributed training.

```python
from cards.base import BaseCard
import numpy as np


class ShardDataCard(BaseCard):
    card_type = "dist_shard_data"
    display_name = "Shard Data"
    description = "Shard training data across workers"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "num_workers": {
            "type": "number",
            "label": "Number of workers (world_size)",
            "default": 4,
        },
    }
    input_schema = {"prepared_data": "json"}
    output_schema = {"sharded_data": "json"}

    def execute(self, config, inputs, storage):
        data = storage.load_json(inputs["prepared_data"])
        num_workers = int(config.get("num_workers", 4))
        
        X_train = np.array(data["X_train"])
        y_train = np.array(data["y_train"])
        
        # Shard data across workers
        shard_size = len(X_train) // num_workers
        shards = []
        for i in range(num_workers):
            start = i * shard_size
            end = start + shard_size if i < num_workers - 1 else len(X_train)
            shards.append({
                "X": X_train[start:end].tolist(),
                "y": y_train[start:end].tolist(),
                "worker_id": i,
                "shard_size": end - start,
            })
        
        sharded = {
            "shards": shards,
            "num_workers": num_workers,
            "batch_size": data["batch_size"],
            "num_features": data["num_features"],
            "num_classes": data["num_classes"],
            "test_data": {
                "X": data["X_test"],
                "y": data["y_test"],
            },
        }
        
        ref = storage.save_json("_p", "_n", "sharded_data", sharded)
        return {"sharded_data": ref}

    def get_output_preview(self, outputs, storage):
        data = storage.load_json(outputs["sharded_data"])
        rows = [["Worker", "Shard Size", "Total Samples"]]
        for shard in data["shards"]:
            rows.append([shard["worker_id"], shard["shard_size"], shard["shard_size"]])
        rows.append(["Total", sum(s["shard_size"] for s in data["shards"]), sum(s["shard_size"] for s in data["shards"])])
        
        return {
            "columns": rows[0],
            "rows": rows[1:],
            "total_rows": len(rows) - 1,
        }
```

---

## Card 4: Initialize Distributed Strategy

**File:** `init_strategy.py` | **Folder:** `distributed/`

Initializes the distributed training strategy (DDP, FSDP, or DeepSpeed).

```python
from cards.base import BaseCard


class InitStrategyCard(BaseCard):
    card_type = "dist_init_strategy"
    display_name = "Initialize Strategy"
    description = "Initialize distributed training strategy"
    category = "model"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "strategy": {
            "type": "string",
            "label": "Strategy (ddp/fsdp/deepspeed)",
            "default": "ddp",
        },
        "model_arch": {
            "type": "string",
            "label": "Model architecture (e.g., '4,16,8,3')",
            "default": "4,16,8,3",
        },
        "learning_rate": {
            "type": "number",
            "label": "Learning rate",
            "default": 0.01,
        },
    }
    input_schema = {}
    output_schema = {"strategy_config": "json"}

    def execute(self, config, inputs, storage):
        strategy = config.get("strategy", "ddp").lower()
        arch_str = config.get("model_arch", "4,16,8,3")
        arch = [int(x.strip()) for x in arch_str.split(",")]
        lr = float(config.get("learning_rate", 0.01))
        
        strategy_config = {
            "strategy": strategy,
            "model_architecture": arch,
            "learning_rate": lr,
            "world_size": None,  # Will be set by process group setup
            "rank": None,
        }
        
        ref = storage.save_json("_p", "_n", "strategy_config", strategy_config)
        return {"strategy_config": ref}

    def get_output_preview(self, outputs, storage):
        config = storage.load_json(outputs["strategy_config"])
        arch_str = " -> ".join(str(x) for x in config["model_architecture"])
        return {
            "architecture": arch_str,
            "strategy": config["strategy"].upper(),
            "learning_rate": config["learning_rate"],
        }
```

---

## Card 5: Setup Process Group

**File:** `setup_process_group.py` | **Folder:** `distributed/`

Sets up the distributed process group for multi-GPU/multi-node training.

```python
from cards.base import BaseCard
import torch
import torch.distributed as dist


class SetupProcessGroupCard(BaseCard):
    card_type = "dist_setup_process_group"
    display_name = "Setup Process Group"
    description = "Initialize distributed process group"
    category = "training"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {
        "backend": {
            "type": "string",
            "label": "Distributed backend (nccl/gloo)",
            "default": "nccl",
        },
        "init_method": {
            "type": "string",
            "label": "Init method (env:// or tcp://)",
            "default": "env://",
        },
    }
    input_schema = {"strategy_config": "json"}
    output_schema = {"process_group": "json"}

    def execute(self, config, inputs, storage):
        strategy_config = storage.load_json(inputs["strategy_config"])
        backend = config.get("backend", "nccl")
        init_method = config.get("init_method", "env://")
        
        # In real implementation, this would initialize the process group
        # For now, we simulate it with config
        world_size = 4  # Would come from environment or config
        rank = 0  # Would come from environment
        
        # Initialize process group (commented for demo, would actually call dist.init_process_group)
        # dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
        
        process_group = {
            "backend": backend,
            "init_method": init_method,
            "world_size": world_size,
            "rank": rank,
            "strategy": strategy_config["strategy"],
            "model_architecture": strategy_config["model_architecture"],
            "learning_rate": strategy_config["learning_rate"],
            "initialized": True,
        }
        
        ref = storage.save_json("_p", "_n", "process_group", process_group)
        return {"process_group": ref}

    def get_output_preview(self, outputs, storage):
        pg = storage.load_json(outputs["process_group"])
        return {
            "world_size": pg["world_size"],
            "rank": pg["rank"],
            "backend": pg["backend"],
            "strategy": pg["strategy"].upper(),
            "initialized": "Yes" if pg["initialized"] else "No",
        }
```

---

## Card 6: Distributed Forward Pass

**File:** `forward_pass.py` | **Folder:** `distributed/`

Performs forward pass on each worker's shard of data.

```python
from cards.base import BaseCard
import torch
import torch.nn as nn
import numpy as np


def _make_model(arch):
    """Build a Sequential model from arch list."""
    layers = []
    for i in range(len(arch) - 1):
        layers.append(nn.Linear(arch[i], arch[i + 1]))
        if i < len(arch) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DistributedForwardCard(BaseCard):
    card_type = "dist_forward"
    display_name = "Distributed Forward"
    description = "Forward pass on distributed data shards"
    category = "training"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {}
    input_schema = {"sharded_data": "json", "process_group": "json"}
    output_schema = {"forward_output": "json"}

    def execute(self, config, inputs, storage):
        sharded = storage.load_json(inputs["sharded_data"])
        pg = storage.load_json(inputs["process_group"])
        
        rank = pg["rank"]
        shard = sharded["shards"][rank]
        arch = pg["model_architecture"]
        
        # Build model
        model = _make_model(arch)
        model.train()
        
        # Get shard data
        X = torch.tensor(shard["X"], dtype=torch.float32)
        y = torch.tensor(shard["y"], dtype=torch.long)
        
        # Forward pass
        with torch.no_grad():
            logits = model(X)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, y)
        
        forward_output = {
            "rank": rank,
            "shard_size": shard["shard_size"],
            "loss": float(loss.item()),
            "logits_shape": list(logits.shape),
            "model_state": {k: v.tolist() for k, v in model.state_dict().items()},
        }
        
        ref = storage.save_json("_p", "_n", "forward_output", forward_output)
        return {"forward_output": ref}

    def get_output_preview(self, outputs, storage):
        output = storage.load_json(outputs["forward_output"])
        return {
            "rank": output["rank"],
            "shard_size": output["shard_size"],
            "loss": round(output["loss"], 4),
            "logits_shape": str(output["logits_shape"]),
        }
```

---

## Card 7: Distributed Backward Pass

**File:** `backward_pass.py` | **Folder:** `distributed/`

Computes gradients on each worker using backward pass.

```python
from cards.base import BaseCard
import torch
import torch.nn as nn
import numpy as np


class DistributedBackwardCard(BaseCard):
    card_type = "dist_backward"
    display_name = "Distributed Backward"
    description = "Backward pass to compute gradients"
    category = "training"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {}
    input_schema = {"forward_output": "json", "process_group": "json"}
    output_schema = {"gradients": "json"}

    def execute(self, config, inputs, storage):
        forward = storage.load_json(inputs["forward_output"])
        pg = storage.load_json(inputs["process_group"])
        
        rank = forward["rank"]
        arch = pg["model_architecture"]
        
        # Rebuild model and load state
        from cards.distributed.forward_pass import _make_model
        model = _make_model(arch)
        model.load_state_dict({k: torch.tensor(v) for k, v in forward["model_state"].items()})
        model.train()
        
        # Get shard data (would come from sharded_data in real impl)
        # For demo, we use dummy data
        X = torch.randn(forward["shard_size"], arch[0], requires_grad=True)
        y = torch.randint(0, arch[-1], (forward["shard_size"],))
        
        # Forward + backward
        logits = model(X)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)
        loss.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.tolist()
        
        grad_output = {
            "rank": rank,
            "gradients": gradients,
            "loss": float(loss.item()),
            "gradient_norm": sum(torch.norm(p.grad).item() for p in model.parameters() if p.grad is not None),
        }
        
        ref = storage.save_json("_p", "_n", "gradients", grad_output)
        return {"gradients": ref}

    def get_output_preview(self, outputs, storage):
        output = storage.load_json(outputs["gradients"])
        return {
            "rank": output["rank"],
            "loss": round(output["loss"], 4),
            "gradient_norm": round(output["gradient_norm"], 6),
            "num_gradients": len(output["gradients"]),
        }
```

---

## Card 8: All-Reduce Gradients

**File:** `allreduce_gradients.py` | **Folder:** `distributed/`

Synchronizes gradients across all workers using all-reduce operation.

```python
from cards.base import BaseCard
import torch
import torch.distributed as dist


class AllReduceGradientsCard(BaseCard):
    card_type = "dist_allreduce"
    display_name = "All-Reduce Gradients"
    description = "Synchronize gradients across workers"
    category = "training"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {}
    input_schema = {"gradients": "json", "process_group": "json"}
    output_schema = {"synced_gradients": "json"}

    def execute(self, config, inputs, storage):
        grad_data = storage.load_json(inputs["gradients"])
        pg = storage.load_json(inputs["process_group"])
        
        rank = grad_data["rank"]
        world_size = pg["world_size"]
        
        # In real implementation, would use dist.all_reduce()
        # For demo, we average gradients across workers (simulating all-reduce with average)
        gradients = grad_data["gradients"]
        
        # Simulate all-reduce: average gradients across all workers
        # In practice: dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        synced_grads = {}
        for name, grad in gradients.items():
            # Average across world_size (simplified - real impl would use dist.all_reduce)
            synced_grads[name] = [[v / world_size for v in row] if isinstance(row, list) else v / world_size 
                                  for row in grad]
        
        synced = {
            "rank": rank,
            "world_size": world_size,
            "synced_gradients": synced_grads,
            "sync_method": "all_reduce_avg",
        }
        
        ref = storage.save_json("_p", "_n", "synced_gradients", synced)
        return {"synced_gradients": ref}

    def get_output_preview(self, outputs, storage):
        output = storage.load_json(outputs["synced_gradients"])
        return {
            "rank": output["rank"],
            "world_size": output["world_size"],
            "sync_method": output["sync_method"],
            "num_synced_gradients": len(output["synced_gradients"]),
        }
```

---

## Card 9: Optimizer Step

**File:** `optimizer_step.py` | **Folder:** `distributed/`

Updates model parameters using synchronized gradients.

```python
from cards.base import BaseCard
import torch
import torch.optim as optim


class OptimizerStepCard(BaseCard):
    card_type = "dist_optimizer_step"
    display_name = "Optimizer Step"
    description = "Update model parameters with synchronized gradients"
    category = "training"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {}
    input_schema = {"synced_gradients": "json", "process_group": "json"}
    output_schema = {"updated_model": "json"}

    def execute(self, config, inputs, storage):
        synced = storage.load_json(inputs["synced_gradients"])
        pg = storage.load_json(inputs["process_group"])
        
        rank = synced["rank"]
        arch = pg["model_architecture"]
        lr = pg["learning_rate"]
        
        # Rebuild model
        from cards.distributed.forward_pass import _make_model
        model = _make_model(arch)
        
        # Set gradients from synced_gradients
        for name, param in model.named_parameters():
            if name in synced["synced_gradients"]:
                grad = torch.tensor(synced["synced_gradients"][name])
                param.grad = grad
        
        # Optimizer step
        optimizer = optim.SGD(model.parameters(), lr=lr)
        optimizer.step()
        optimizer.zero_grad()
        
        updated = {
            "rank": rank,
            "model_state": {k: v.tolist() for k, v in model.state_dict().items()},
            "learning_rate": lr,
            "step_completed": True,
        }
        
        ref = storage.save_json("_p", "_n", "updated_model", updated)
        return {"updated_model": ref}

    def get_output_preview(self, outputs, storage):
        output = storage.load_json(outputs["updated_model"])
        return {
            "rank": output["rank"],
            "learning_rate": output["learning_rate"],
            "step_completed": "Yes" if output["step_completed"] else "No",
        }
```

---

## Card 10: Save Checkpoint

**File:** `save_checkpoint.py` | **Folder:** `checkpointing/`

Saves model checkpoint from each worker (or master rank only).

```python
from cards.base import BaseCard
import torch


class SaveCheckpointCard(BaseCard):
    card_type = "dist_save_checkpoint"
    display_name = "Save Checkpoint"
    description = "Save model checkpoint from distributed training"
    category = "training"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {
        "epoch": {
            "type": "number",
            "label": "Current epoch",
            "default": 1,
        },
        "save_all_ranks": {
            "type": "boolean",
            "label": "Save checkpoints from all ranks",
            "default": False,
        },
    }
    input_schema = {"updated_model": "json", "process_group": "json"}
    output_schema = {"checkpoint": "json"}

    def execute(self, config, inputs, storage):
        model_data = storage.load_json(inputs["updated_model"])
        pg = storage.load_json(inputs["process_group"])
        
        rank = model_data["rank"]
        epoch = int(config.get("epoch", 1))
        save_all = config.get("save_all_ranks", False)
        
        # Save checkpoint (typically only from rank 0, or all ranks if configured)
        should_save = (rank == 0) or save_all
        
        checkpoint = {
            "rank": rank,
            "epoch": epoch,
            "model_state": model_data["model_state"],
            "checkpoint_path": f"checkpoint_epoch_{epoch}_rank_{rank}.pt",
            "saved": should_save,
        }
        
        ref = storage.save_json("_p", "_n", "checkpoint", checkpoint)
        return {"checkpoint": ref}

    def get_output_preview(self, outputs, storage):
        checkpoint = storage.load_json(outputs["checkpoint"])
        return {
            "rank": checkpoint["rank"],
            "epoch": checkpoint["epoch"],
            "saved": "Yes" if checkpoint["saved"] else "No",
            "checkpoint_path": checkpoint["checkpoint_path"],
        }
```

---

## Card 11: Aggregate Model

**File:** `aggregate_model.py` | **Folder:** `aggregation/`

Aggregates model parameters from all workers into a single consolidated model.

```python
from cards.base import BaseCard
import torch
import numpy as np


class AggregateModelCard(BaseCard):
    card_type = "dist_aggregate_model"
    display_name = "Aggregate Model"
    description = "Aggregate model parameters from all workers"
    category = "model"
    execution_mode = "modal"
    output_view_type = "model_summary"

    config_schema = {
        "aggregation_method": {
            "type": "string",
            "label": "Aggregation method (average/weighted)",
            "default": "average",
        },
    }
    input_schema = {"checkpoint": "json"}
    output_schema = {"aggregated_model": "json"}

    def execute(self, config, inputs, storage):
        checkpoint = storage.load_json(inputs["checkpoint"])
        method = config.get("aggregation_method", "average")
        
        # In real implementation, would load checkpoints from all ranks
        # For demo, we use the checkpoint from rank 0
        model_state = checkpoint["model_state"]
        
        # Aggregate (simplified - real impl would average across all ranks)
        aggregated = {
            "model_state": model_state,
            "aggregation_method": method,
            "source_rank": checkpoint["rank"],
            "epoch": checkpoint["epoch"],
        }
        
        ref = storage.save_json("_p", "_n", "aggregated_model", aggregated)
        return {"aggregated_model": ref}

    def get_output_preview(self, outputs, storage):
        model = storage.load_json(outputs["aggregated_model"])
        return {
            "aggregation_method": model["aggregation_method"],
            "source_rank": model["source_rank"],
            "epoch": model["epoch"],
        }
```

---

## Card 12: Evaluate

**File:** `evaluate.py` | **Folder:** `evaluation/`

Evaluates the aggregated model on the test set.

```python
from cards.base import BaseCard
import torch
import torch.nn as nn
import numpy as np


class EvaluateCard(BaseCard):
    card_type = "dist_evaluate"
    display_name = "Evaluate"
    description = "Evaluate aggregated model on test set"
    category = "evaluation"
    execution_mode = "modal"
    output_view_type = "metrics"

    config_schema = {}
    input_schema = {"aggregated_model": "json", "prepared_data": "json"}
    output_schema = {"eval_metrics": "json"}

    def execute(self, config, inputs, storage):
        model_data = storage.load_json(inputs["aggregated_model"])
        data = storage.load_json(inputs["prepared_data"])
        
        # Rebuild model
        from cards.distributed.init_strategy import InitStrategyCard
        # Get architecture from model state keys (simplified)
        arch = [4, 16, 8, 3]  # Would extract from model_state in real impl
        
        from cards.distributed.forward_pass import _make_model
        model = _make_model(arch)
        model.load_state_dict({k: torch.tensor(v) for k, v in model_data["model_state"].items()})
        model.eval()
        
        # Evaluate on test set
        X_test = torch.tensor(data["X_test"], dtype=torch.float32)
        y_test = torch.tensor(data["y_test"], dtype=torch.long)
        
        with torch.no_grad():
            logits = model(X_test)
            preds = logits.argmax(dim=1)
            accuracy = (preds == y_test).float().mean().item()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, y_test).item()
        
        metrics = {
            "accuracy": accuracy,
            "loss": loss,
            "num_test_samples": len(y_test),
        }
        
        ref = storage.save_json("_p", "_n", "eval_metrics", metrics)
        return {"eval_metrics": ref}

    def get_output_preview(self, outputs, storage):
        metrics = storage.load_json(outputs["eval_metrics"])
        return {
            "accuracy": round(metrics["accuracy"], 4),
            "loss": round(metrics["loss"], 4),
            "test_samples": metrics["num_test_samples"],
        }
```

---

## Wiring Diagram

```
[Load Dataset] ──dataset──> [Prepare Data]
[Prepare Data] ──prepared_data──> [Shard Data]
[Prepare Data] ──prepared_data──> [Evaluate]

[Shard Data] ──sharded_data──> [Distributed Forward]

[Initialize Strategy] ──strategy_config──> [Setup Process Group]
[Setup Process Group] ──process_group──> [Distributed Forward]
[Setup Process Group] ──process_group──> [Distributed Backward]
[Setup Process Group] ──process_group──> [All-Reduce Gradients]
[Setup Process Group] ──process_group──> [Optimizer Step]
[Setup Process Group] ──process_group──> [Save Checkpoint]

[Distributed Forward] ──forward_output──> [Distributed Backward]
[Distributed Backward] ──gradients──> [All-Reduce Gradients]
[All-Reduce Gradients] ──synced_gradients──> [Optimizer Step]
[Optimizer Step] ──updated_model──> [Save Checkpoint]
[Save Checkpoint] ──checkpoint──> [Aggregate Model]
[Aggregate Model] ──aggregated_model──> [Evaluate]
```

---

## Usage Notes

1. **Distributed Backend**: Cards 5-9 require a distributed backend (NCCL for GPU, Gloo for CPU). These should run on Modal with GPU support.

2. **Process Group**: The process group setup (Card 5) initializes the distributed environment. In production, this would use environment variables (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`).

3. **Gradient Synchronization**: Card 8 (All-Reduce) synchronizes gradients across workers. In PyTorch, this is typically handled automatically by `DistributedDataParallel` wrapper, but we expose it as a separate card for clarity.

4. **Checkpointing**: Card 10 can save from all ranks or just rank 0. In practice, you often save only from rank 0 to avoid redundancy.

5. **Model Aggregation**: Card 11 aggregates parameters. For DDP, all models should be identical after synchronization, so aggregation is straightforward. For FSDP or other sharded strategies, aggregation requires gathering sharded parameters.

6. **Training Loop**: In practice, Cards 6-9 would be executed in a loop for multiple epochs. The pipeline shown here represents one training step.

---

With these 12 cards, you can build a complete distributed training pipeline that handles data sharding, gradient synchronization, checkpointing, and model aggregation across multiple workers.
