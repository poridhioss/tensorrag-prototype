# Training Loop Components Guide

## Overview

The training loop has been broken down into individual card components, giving you granular control over each step of the training process. This allows you to customize the training loop, add custom operations between steps, or modify the flow as needed.

## Training Loop Steps

The standard training loop consists of 5 main steps:

1. **Zero Gradients** - Clear previous gradients
2. **Forward Pass** - Compute model predictions
3. **Calculate Loss** - Compute loss between predictions and targets
4. **Backward Pass** - Compute gradients via backpropagation
5. **Optimizer Step** - Update model parameters

## Available Cards

### 1. Prepare Batch Card
**Card Type**: `training_prepare_batch`
- **Purpose**: Prepare a batch of data for training
- **Inputs**: 
  - `train_dataset`: Training dataset (DataFrame)
- **Outputs**:
  - `batch_data`: Serialized batch tensors (X and y)
  - `batch_info`: Batch metadata
- **Configuration**:
  - `batch_index`: Which batch to retrieve (default: 0)
  - `batch_size`: Size of each batch (default: 32)

### 2. Initialize Optimizer Card
**Card Type**: `training_init_optimizer`
- **Purpose**: Initialize optimizer for model training
- **Inputs**:
  - `model`: Model to optimize
- **Outputs**:
  - `optimizer`: Initialized optimizer
- **Configuration**:
  - `optimizer_type`: "adam", "sgd", or "rmsprop" (default: "adam")
  - `learning_rate`: Learning rate (default: 0.001)
  - `weight_decay`: L2 regularization (default: 0.0)

### 3. Zero Gradients Card
**Card Type**: `training_zero_grad`
- **Purpose**: Zero out gradients in the optimizer
- **Inputs**:
  - `optimizer`: Optimizer with gradients to clear
- **Outputs**:
  - `optimizer`: Optimizer with zeroed gradients
- **Configuration**: None

### 4. Forward Pass Card
**Card Type**: `training_forward`
- **Purpose**: Perform forward pass through the model
- **Inputs**:
  - `model`: Model to run forward pass on
  - `batch_data`: Batch data (X and y tensors)
- **Outputs**:
  - `predictions`: Model predictions
  - `model`: Model state (preserved)
- **Configuration**:
  - `batch_index`: Batch index for tracking (default: 0)

### 5. Calculate Loss Card
**Card Type**: `training_loss`
- **Purpose**: Calculate loss between predictions and targets
- **Inputs**:
  - `predictions`: Model predictions
  - `batch_data`: Batch data containing targets
- **Outputs**:
  - `loss`: Loss tensor (with gradients)
  - `loss_value`: Loss value (scalar)
- **Configuration**:
  - `loss_type`: "mse", "mae", or "huber" (default: "mse")

### 6. Backward Pass Card
**Card Type**: `training_backward`
- **Purpose**: Compute gradients via backpropagation
- **Inputs**:
  - `loss`: Loss tensor
  - `model`: Model to compute gradients for
- **Outputs**:
  - `model`: Model with computed gradients
  - `gradients_computed`: Status information
- **Configuration**:
  - `retain_graph`: Retain computation graph (default: false)

### 7. Optimizer Step Card
**Card Type**: `training_optimizer_step`
- **Purpose**: Update model parameters using optimizer
- **Inputs**:
  - `optimizer`: Optimizer
  - `model`: Model with gradients
- **Outputs**:
  - `model`: Updated model
  - `optimizer`: Updated optimizer
  - `step_info`: Step information
- **Configuration**:
  - `gradient_clip`: Gradient clipping value (0 = no clipping, default: 0)

## Example Pipeline

### Basic Training Loop

```
Model Define (GPU)
    ↓
Initialize Optimizer
    ↓
Prepare Batch (batch_index=0)
    ↓
Zero Gradients → Forward Pass → Calculate Loss → Backward Pass → Optimizer Step
    ↓
Prepare Batch (batch_index=1)
    ↓
Zero Gradients → Forward Pass → Calculate Loss → Backward Pass → Optimizer Step
    ↓
... (repeat for all batches)
```

### Custom Training Loop

You can customize the loop by:
- Adding custom operations between steps
- Skipping certain steps conditionally
- Adding multiple loss calculations
- Using different optimizers for different parts

Example: Custom loss with regularization
```
Forward Pass
    ↓
Calculate Loss (MSE)
    ↓
[Custom Card: Add L1 Regularization]
    ↓
Backward Pass
    ↓
Optimizer Step
```

## Connection Flow

### Standard Training Step

1. **Zero Gradients** receives `optimizer` → outputs `optimizer`
2. **Forward Pass** receives `model` and `batch_data` → outputs `predictions` and `model`
3. **Calculate Loss** receives `predictions` and `batch_data` → outputs `loss`
4. **Backward Pass** receives `loss` and `model` → outputs `model` (with gradients)
5. **Optimizer Step** receives `optimizer` and `model` → outputs `model` and `optimizer`

### State Management

- **Model state** is preserved through the chain
- **Optimizer state** is preserved through the chain
- **Batch data** can be reused (forward pass and loss calculation both need it)
- Each card saves its outputs to storage, allowing state to persist between executions

## Advanced Usage

### Multiple Batches

To train on multiple batches, you can:
1. Create multiple Prepare Batch cards with different `batch_index` values
2. Chain the training steps for each batch
3. Connect the final model/optimizer to the next batch's training steps

### Custom Loss Functions

You can create custom cards that:
- Combine multiple loss functions
- Add regularization terms
- Implement custom loss calculations

### Gradient Accumulation

For gradient accumulation:
1. Run Forward Pass → Loss → Backward Pass multiple times (without optimizer step)
2. Accumulate gradients in the model
3. Run Optimizer Step once after accumulating

### Learning Rate Scheduling

You can add custom cards that:
- Modify optimizer learning rate between steps
- Implement learning rate schedules
- Add warmup periods

## Tips

1. **State Preservation**: Make sure to connect model and optimizer outputs to the next step's inputs
2. **Batch Data**: The same batch_data can be used for both forward pass and loss calculation
3. **Device Management**: Models are automatically moved to CPU for storage, then back to GPU when loaded
4. **Gradient Tracking**: Use `retain_graph=True` in backward pass if you need to backpropagate multiple times

## Limitations

- Currently, each card execution is independent (runs in separate Modal containers)
- Model and optimizer state must be serialized/deserialized between cards
- This adds overhead compared to a single training loop
- Best for experimentation and custom training logic

## Future Enhancements

Potential improvements:
- State caching to reduce serialization overhead
- Batch iterator card for automatic batch management
- Training loop wrapper card for convenience
- Gradient accumulation support
- Learning rate scheduler cards
