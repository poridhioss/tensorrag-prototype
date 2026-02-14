# Training Card Usage Guide

## When to Use Which Approach

### Use **Full Train Card** (Recommended for Most Cases)
- **When**: You want a complete, automated training loop
- **Cards**: `Train` (CPU) or `Train (GPU)` (GPU)
- **Pros**: 
  - Simple and fast to set up
  - Handles the entire training loop automatically
  - Less error-prone
  - Better performance (single execution)
- **Use Case**: Standard training workflows, quick prototyping

### Use **Individual Training Steps** (Advanced Control)
- **When**: You need granular control over the training process
- **Cards**: All 7 training step cards
- **Pros**:
  - Full control over each step
  - Can add custom operations between steps
  - Can modify the loop structure
  - Educational/debugging purposes
- **Cons**:
  - More complex setup
  - Slower (each step runs separately)
  - More error-prone
- **Use Case**: Custom training logic, experimentation, learning

---

## Order of Individual Training Steps

### Initial Setup (One Time)

1. **Model Define (GPU)**
   - Define your neural network architecture
   - Output: `model_spec`

2. **Initialize Optimizer**
   - Input: `model` (from Model Define)
   - Output: `optimizer`
   - Configure: optimizer type, learning rate, weight decay

### Per-Batch Training Loop (Repeat for Each Batch)

For each batch in your training data, execute these steps **in order**:

#### Step 1: **Prepare Batch**
- Input: `train_dataset` (from Data Split)
- Output: `batch_data`, `batch_info`
- Configure: `batch_index` (0, 1, 2, ...), `batch_size`

#### Step 2: **Zero Gradients**
- Input: `optimizer`
- Output: `optimizer` (with zeroed gradients)
- **Note**: This clears gradients from the previous iteration

#### Step 3: **Forward Pass**
- Input: `model`, `batch_data`
- Output: `predictions`, `model`
- **Note**: Model state is preserved

#### Step 4: **Calculate Loss**
- Input: `predictions`, `batch_data`
- Output: `loss`, `loss_value`
- Configure: `loss_type` (mse, mae, huber)

#### Step 5: **Backward Pass**
- Input: `loss`, `model`
- Output: `model` (with computed gradients), `gradients_computed`
- **Note**: Gradients are now stored in model parameters

#### Step 6: **Optimizer Step**
- Input: `optimizer`, `model`
- Output: `model` (updated), `optimizer` (updated), `step_info`
- Configure: `gradient_clip` (optional)
- **Note**: Model parameters are updated here

### Repeat Steps 1-6 for Each Batch

After Step 6, you can:
- Go back to Step 1 with the next `batch_index`
- Or use the updated `model` and `optimizer` for the next iteration

---

## Complete Pipeline Example

### Using Full Train Card (Simple)

```
Data Load
    ↓
Data Split
    ↓
Model Define (GPU)
    ↓
Train (GPU)  ← Single card handles everything
    ↓
Evaluate
```

### Using Individual Steps (Advanced)

```
Data Load
    ↓
Data Split
    ↓
Model Define (GPU)
    ↓
Initialize Optimizer
    ↓
[Loop for each batch:]
    Prepare Batch (batch_index=0)
        ↓
    Zero Gradients
        ↓
    Forward Pass
        ↓
    Calculate Loss
        ↓
    Backward Pass
        ↓
    Optimizer Step
        ↓
    [Connect back to Prepare Batch with batch_index=1]
    [Repeat until all batches processed]
    ↓
Evaluate
```

---

## Important Notes

### State Management
- **Model state** must flow through: Forward Pass → Backward Pass → Optimizer Step
- **Optimizer state** must flow through: Zero Gradients → Optimizer Step
- **Batch data** is used in: Forward Pass and Calculate Loss

### Connection Rules
1. **Model** flows: Forward Pass → Backward Pass → Optimizer Step → (back to Forward Pass for next batch)
2. **Optimizer** flows: Zero Gradients → Optimizer Step → (back to Zero Gradients for next batch)
3. **Batch Data** flows: Prepare Batch → Forward Pass AND Calculate Loss

### Typical Training Loop Structure

For **one epoch** (all batches):
```
For batch_index = 0 to num_batches:
    1. Prepare Batch (batch_index)
    2. Zero Gradients
    3. Forward Pass
    4. Calculate Loss
    5. Backward Pass
    6. Optimizer Step
```

For **multiple epochs**, you'd need to:
- Repeat the entire loop for each epoch
- Or create a more complex pipeline structure

---

## Recommendation

**For most users**: Use the **Train (GPU)** card - it's simpler and handles everything automatically.

**Use individual steps when**:
- You need custom loss functions
- You want to add operations between steps
- You're learning how training works
- You need gradient accumulation
- You want to implement custom training logic

---

## Quick Reference: Step Order

```
1. Prepare Batch
2. Zero Gradients
3. Forward Pass
4. Calculate Loss
5. Backward Pass
6. Optimizer Step
```

**Repeat steps 1-6 for each batch in your dataset.**
