# Training Flow - Single Connection Per Output Handle

## Constraint
Each output handle can connect to **one input handle only**. Cards with multiple outputs provide separate handles.

---

## Corrected Complete Flow

### Setup Phase

```
Model Define (GPU)
    ↓ model_spec
Build Model
    ↓ model
Initialize Optimizer
    ├─→ optimizer (separate output handle)
    └─→ model (separate output handle)
```

**Connections:**
1. Model Define (GPU) → Build Model (`model_spec`)
2. Build Model → Initialize Optimizer (`model`)
3. Initialize Optimizer → Zero Gradients (`optimizer` - separate handle)
4. Initialize Optimizer → Forward Pass (`model` - separate handle)

---

### Training Loop (One Batch)

```
Prepare Batch
    ├─→ batch_data (handle 1) → Forward Pass
    └─→ batch_data (handle 2) → Calculate Loss
         (Note: Create TWO separate edges from Prepare Batch)

Zero Gradients
    └─→ optimizer → Optimizer Step

Forward Pass
    ├─→ predictions → Calculate Loss
    └─→ model → Backward Pass

Calculate Loss
    └─→ loss → Backward Pass

Backward Pass
    └─→ model → Optimizer Step

Optimizer Step
    ├─→ model → Forward Pass (next batch)
    └─→ optimizer → Zero Gradients (next batch)
```

---

## Visual Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    SETUP                                 │
└─────────────────────────────────────────────────────────┘

Model Define (GPU)
    ↓
Build Model
    ↓
Initialize Optimizer
    ├─→ [optimizer output] → Zero Gradients
    └─→ [model output] → Forward Pass

┌─────────────────────────────────────────────────────────┐
│              TRAINING LOOP (Batch 0)                    │
└─────────────────────────────────────────────────────────┘

Prepare Batch (index: 0)
    ├─→ [batch_data output 1] → Forward Pass
    └─→ [batch_data output 2] → Calculate Loss
         (Create 2 separate connections)

Zero Gradients → Optimizer Step (optimizer)

Forward Pass
    ├─→ [predictions output] → Calculate Loss
    └─→ [model output] → Backward Pass

Calculate Loss → Backward Pass (loss)

Backward Pass → Optimizer Step (model)

Optimizer Step
    ├─→ [model output] → Forward Pass (next batch)
    └─→ [optimizer output] → Zero Gradients (next batch)

┌─────────────────────────────────────────────────────────┐
│              TRAINING LOOP (Batch 1)                     │
└─────────────────────────────────────────────────────────┘

Prepare Batch (index: 1)
    ├─→ Forward Pass
    └─→ Calculate Loss

[Same connections as Batch 0, but with updated model/optimizer]

... Repeat for all batches ...
```

---

## Key Cards with Multiple Outputs

### Initialize Optimizer
- **Output 1:** `optimizer` → connects to Zero Gradients
- **Output 2:** `model` → connects to Forward Pass

### Forward Pass
- **Output 1:** `predictions` → connects to Calculate Loss
- **Output 2:** `model` → connects to Backward Pass

### Optimizer Step
- **Output 1:** `model` → connects to Forward Pass (next batch)
- **Output 2:** `optimizer` → connects to Zero Gradients (next batch)

### Prepare Batch
- **Output 1:** `batch_data` → connects to Forward Pass
- **Output 2:** `batch_data` → connects to Calculate Loss
- **Note:** You need to create TWO separate edges from the same `batch_data` output handle

---

## Connection Checklist

### Setup
- [ ] Model Define (GPU) → Build Model
- [ ] Build Model → Initialize Optimizer
- [ ] Initialize Optimizer `optimizer` → Zero Gradients
- [ ] Initialize Optimizer `model` → Forward Pass

### Per Batch
- [ ] Prepare Batch `batch_data` → Forward Pass (edge 1)
- [ ] Prepare Batch `batch_data` → Calculate Loss (edge 2)
- [ ] Zero Gradients → Optimizer Step
- [ ] Forward Pass `predictions` → Calculate Loss
- [ ] Forward Pass `model` → Backward Pass
- [ ] Calculate Loss → Backward Pass
- [ ] Backward Pass → Optimizer Step
- [ ] Optimizer Step `model` → Forward Pass (next batch)
- [ ] Optimizer Step `optimizer` → Zero Gradients (next batch)

---

## Important Notes

1. **Multiple Output Handles**: Cards like Initialize Optimizer, Forward Pass, and Optimizer Step have multiple output handles - each can connect to one input
2. **Prepare Batch Special Case**: The `batch_data` output needs to go to TWO places. You'll create two separate edges from the same output handle
3. **Sequential Flow**: Model flows: Build Model → Initialize Optimizer → Forward Pass → Backward Pass → Optimizer Step → Forward Pass (loop)
4. **Optimizer Flow**: Optimizer flows: Initialize Optimizer → Zero Gradients → Optimizer Step → Zero Gradients (loop)
