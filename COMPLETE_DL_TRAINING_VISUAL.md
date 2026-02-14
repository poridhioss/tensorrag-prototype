# Complete Deep Learning Training Pipeline - Visual Guide

## Full Pipeline Overview

This guide shows the complete visual flow for training a deep learning model using individual training step cards.

---

## Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PREPARATION                                │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  Data Load   │
│  (sample:    │
│  boston_     │
│  housing)    │
└──────┬───────┘
       │ dataset
       ↓
┌──────────────┐
│ Data Split   │
│ (80/20)      │
└──────┬───────┘
       │ train_dataset
       │ test_dataset
       │
       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          MODEL SETUP                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│ Model Define (GPU)  │
│                     │
│ - model_type:       │
│   deep_neural_      │
│   network           │
│ - hidden_layers:    │
│   [128, 64, 32]     │
│ - activation: relu   │
│ - dropout: 0.2      │
│ - learning_rate:    │
│   0.001             │
│ - epochs: 50        │
│ - batch_size: 32    │
└──────────┬──────────┘
           │ model_spec
           ↓
┌─────────────────────┐
│    Build Model      │
│                     │
│ - input_size: 13    │
│   (number of        │
│   features)         │
└──────────┬──────────┘
           │ model
           ├─────────────────┐
           │                 │
           ↓                 ↓
┌─────────────────────┐ ┌──────────────┐
│ Initialize Optimizer│ │Forward Pass  │
│                     │ │ (1st batch)  │
│ - optimizer_type:   │ │              │
│   adam              │ │              │
│ - learning_rate:    │ │              │
│   0.001             │ │              │
│ - weight_decay: 0.0 │ │              │
└──────────┬──────────┘ └──────────────┘
           │ optimizer
           │
           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (Repeat for Each Batch)                 │
└─────────────────────────────────────────────────────────────────────────┘

           ┌─────────────────┐
           │  Prepare Batch  │
           │                 │
           │ - batch_index: 0│
           │ - batch_size: 32│
           └────────┬────────┘
                    │ batch_data
           ┌────────┴────────┐
           │                 │
           ↓                 ↓
    ┌───────────────┐  ┌──────────────┐
    │ Zero Gradients│  │Forward Pass  │
    │               │  │              │
    │ (clears prev  │  │ - model      │
    │  gradients)   │  │ - batch_data │
    └───────┬───────┘  └──────┬───────┘
            │ optimizer       │ predictions
            │                 │ model
            │                 │
            │                 ↓
            │            ┌──────────────┐
            │            │Calculate Loss│
            │            │              │
            │            │ - loss_type: │
            │            │   mse         │
            │            │ - predictions │
            │            │ - batch_data  │
            │            └──────┬───────┘
            │                  │ loss
            │                  │
            │                  ↓
            │            ┌──────────────┐
            │            │Backward Pass │
            │            │              │
            │            │ - loss       │
            │            │ - model      │
            │            │              │
            │            │ (computes    │
            │            │  gradients)   │
            │            └──────┬───────┘
            │                  │ model
            │                  │ (with gradients)
            │                  │
            └──────────────────┘
                     │
                     ↓
            ┌─────────────────┐
            │ Optimizer Step   │
            │                 │
            │ - optimizer     │
            │ - model         │
            │ - gradient_clip:│
            │   0 (optional)  │
            │                 │
            │ (updates model  │
            │  parameters)    │
            └────────┬────────┘
                     │ model (updated)
                     │ optimizer (updated)
                     │
                     └──→ [Loop back to Prepare Batch]
                          (batch_index: 1, 2, 3, ...)
                          until all batches processed
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION                                      │
└─────────────────────────────────────────────────────────────────────────┘

                     ┌──────────────┐
                     │   Evaluate   │
                     │              │
                     │ - trained_   │
                     │   model      │
                     │ - test_      │
                     │   dataset    │
                     └──────────────┘
```

---

## Step-by-Step Visual Connection Guide

### Phase 1: Data Preparation

```
┌──────────────┐
│  Data Load   │
└──────┬───────┘
       │ dataset
       ↓
┌──────────────┐
│ Data Split   │
└──────┬───────┘
       │ train_dataset
       │ test_dataset
```

**Connections:**
- Data Load `dataset` → Data Split `dataset`

---

### Phase 2: Model Setup

```
┌─────────────────────┐
│ Model Define (GPU)  │
└──────────┬──────────┘
           │ model_spec
           ↓
┌─────────────────────┐
│    Build Model      │
└──────────┬──────────┘
           │ model
           ↓
┌─────────────────────┐
│ Initialize Optimizer│
└──────────┬──────────┘
           │ optimizer
           │
           │ (model flows through to Forward Pass)
```

**Connections:**
1. Model Define (GPU) `model_spec` → Build Model `model_spec`
2. Build Model `model` → Initialize Optimizer `model`
3. Initialize Optimizer `model` → Forward Pass `model` (for first iteration)

**Note:** Since each card has one input and one output connection per handle, the model flows sequentially: Build Model → Initialize Optimizer → Forward Pass

---

### Phase 3: Training Loop (One Batch)

```
                    ┌─────────────────┐
                    │  Prepare Batch  │
                    │  (index: 0)     │
                    └────────┬────────┘
                             │ batch_data
                    ┌────────┴────────┐
                    │                 │
                    ↓                 ↓
            ┌───────────────┐  ┌──────────────┐
            │ Zero Gradients│  │Forward Pass  │
            └───────┬───────┘  └──────┬───────┘
                    │ optimizer       │ predictions
                    │                 │ model
                    │                 │
                    │                 ↓
                    │            ┌──────────────┐
                    │            │Calculate Loss│
                    │            └──────┬───────┘
                    │                  │ loss
                    │                  │
                    │                  ↓
                    │            ┌──────────────┐
                    │            │Backward Pass │
                    │            └──────┬───────┘
                    │                  │ model
                    │                  │
                    └──────────────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │ Optimizer Step   │
                    └────────┬────────┘
                             │ model (updated)
                             │ optimizer (updated)
```

**Connections for One Batch:**
1. Prepare Batch `batch_data` → Forward Pass `batch_data`
2. Prepare Batch `batch_data` → Calculate Loss `batch_data`
3. Zero Gradients `optimizer` → Optimizer Step `optimizer`
4. Forward Pass `predictions` → Calculate Loss `predictions`
5. Forward Pass `model` → Backward Pass `model`
6. Calculate Loss `loss` → Backward Pass `loss`
7. Backward Pass `model` → Optimizer Step `model`

---

### Phase 4: Loop Structure (Multiple Batches)

```
[Batch 0]
Prepare Batch (index: 0) → Zero Grad → Forward → Loss → Backward → Optimizer Step
    │                                                                      │
    │                                                                      │
    └────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
[Batch 1]
Prepare Batch (index: 1) → Zero Grad → Forward → Loss → Backward → Optimizer Step
    │                                                                      │
    │                                                                      │
    └────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
[Batch 2]
Prepare Batch (index: 2) → Zero Grad → Forward → Loss → Backward → Optimizer Step
    │                                                                      │
    │                                                                      │
    └────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
                            [Continue for all batches...]
                                    │
                                    ↓
                            [All batches processed]
                                    │
                                    ↓
```

**Loop Connections:**
- Optimizer Step `model` → Forward Pass `model` (for next batch)
- Optimizer Step `optimizer` → Zero Gradients `optimizer` (for next batch)
- Prepare Batch with next `batch_index`

---

### Phase 5: Evaluation

```
                    ┌──────────────┐
                    │   Evaluate   │
                    │              │
                    │ Inputs:      │
                    │ - trained_   │
                    │   model      │
                    │   (from      │
                    │   Optimizer  │
                    │   Step)      │
                    │ - test_      │
                    │   dataset    │
                    │   (from      │
                    │   Data Split)│
                    └──────────────┘
```

**Connections:**
- Optimizer Step `model` → Evaluate `trained_model`
- Data Split `test_dataset` → Evaluate `test_dataset`

---

## Complete Connection Map

### All Connections in Order

```
1. Data Load → Data Split
   dataset → dataset

2. Model Define (GPU) → Build Model
   model_spec → model_spec

3. Build Model → Initialize Optimizer
   model → model

4. Build Model → Forward Pass (first iteration)
   model → model

5. Data Split → Prepare Batch
   train_dataset → train_dataset

6. Prepare Batch → Forward Pass
   batch_data → batch_data

7. Prepare Batch → Calculate Loss
   batch_data → batch_data

8. Initialize Optimizer → Zero Gradients
   optimizer → optimizer

9. Forward Pass → Calculate Loss
   predictions → predictions

10. Forward Pass → Backward Pass
    model → model

11. Calculate Loss → Backward Pass
    loss → loss

12. Zero Gradients → Optimizer Step
    optimizer → optimizer

13. Backward Pass → Optimizer Step
    model → model

14. Optimizer Step → Forward Pass (next batch)
    model → model

15. Optimizer Step → Zero Gradients (next batch)
    optimizer → optimizer

16. Optimizer Step → Evaluate (after all batches)
    model → trained_model

17. Data Split → Evaluate
    test_dataset → test_dataset
```

---

## Visual Summary: Card Flow

```
DATA PREPARATION:
Data Load → Data Split

MODEL SETUP:
Model Define (GPU) → Build Model → Initialize Optimizer
                    └────────────→ Forward Pass (first)

TRAINING LOOP (per batch):
Prepare Batch ──┬─→ Forward Pass ──┬─→ Calculate Loss ──→ Backward Pass ──→ Optimizer Step
                └─→ Calculate Loss ─┘                                    │
                                                                         │
Zero Gradients ─────────────────────────────────────────────────────────┘
                                                                         │
                                                                         └─→ [Next Batch]

EVALUATION:
Optimizer Step → Evaluate
Data Split ────→ Evaluate
```

---

## Configuration Values Example

### Data Load
- Source: `sample`
- Sample Name: `boston_housing`

### Data Split
- Train Ratio: `0.8`
- Test Ratio: `0.2`

### Model Define (GPU)
- Model Type: `deep_neural_network`
- Hidden Layers: `[128, 64, 32]`
- Activation: `relu`
- Dropout: `0.2`
- Learning Rate: `0.001`
- Epochs: `50`
- Batch Size: `32`

### Build Model
- Input Size: `13` (number of features in boston_housing)

### Initialize Optimizer
- Optimizer Type: `adam`
- Learning Rate: `0.001`
- Weight Decay: `0.0`

### Prepare Batch
- Batch Index: `0` (then 1, 2, 3, ...)
- Batch Size: `32`

### Calculate Loss
- Loss Type: `mse`

### Optimizer Step
- Gradient Clip: `0` (no clipping)

---

## Quick Setup Checklist

### Initial Setup
- [ ] Add Data Load card (configure: sample = boston_housing)
- [ ] Add Data Split card (connect from Data Load)
- [ ] Add Model Define (GPU) card (configure architecture)
- [ ] Add Build Model card (configure: input_size = 13)
- [ ] Add Initialize Optimizer card (configure optimizer)

### Training Loop (for each batch)
- [ ] Add Prepare Batch card (configure: batch_index, batch_size)
- [ ] Add Zero Gradients card
- [ ] Add Forward Pass card
- [ ] Add Calculate Loss card
- [ ] Add Backward Pass card
- [ ] Add Optimizer Step card

### Evaluation
- [ ] Add Evaluate card (connect from Optimizer Step and Data Split)

---

## Tips

1. **Start Simple**: Use the full "Train (GPU)" card first to verify your pipeline works
2. **Then Expand**: Break it down into individual steps for customization
3. **Batch Management**: You'll need to manually create Prepare Batch cards for each batch index
4. **State Flow**: Always connect model and optimizer outputs to the next step's inputs
5. **First Iteration**: Build Model connects to Forward Pass for the first batch
6. **Subsequent Iterations**: Optimizer Step connects back to Forward Pass for next batches

---

## Alternative: Simplified Pipeline (Using Full Train Card)

If you don't need granular control, use this simpler approach:

```
Data Load
    ↓
Data Split
    ↓
Model Define (GPU)
    ↓
Train (GPU)  ← Single card handles entire training loop
    ↓
Evaluate
```

This is much simpler and recommended for most use cases!
