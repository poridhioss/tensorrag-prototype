# Visual Guide: Connecting Training Step Cards

## Overview

This guide shows you **visually** how to connect the individual training step cards together.

---

## Step 1: Setup Phase (One-Time Setup)

```
┌─────────────────────┐
│  Model Define (GPU) │
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
           ↓
      [Ready for training]
```

**What to do:**
1. Drag **Model Define (GPU)** card
2. Drag **Build Model** card
3. Drag **Initialize Optimizer** card
4. Connect: `model_spec` (from Model Define) → `model_spec` (to Build Model)
5. Connect: `model` (from Build Model) → `model` (to Initialize Optimizer)
6. Configure all cards (Build Model needs `input_size` - number of features)

---

## Step 2: Training Loop (Repeat for Each Batch)

### Visual Flow Diagram

```
┌─────────────────┐
│  Prepare Batch  │
│  (batch_index)  │
└─────┬───────────┘
      │ batch_data
      ├─────────────────┐
      │                 │
      ↓                 ↓
┌─────────────┐   ┌──────────────┐
│ Zero Grad   │   │ Forward Pass │
│             │   │              │
└──────┬──────┘   └──────┬───────┘
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
       │                  │ model (with gradients)
       │                  │
       └──────────────────┘
                  │
                  ↓
         ┌─────────────────┐
         │ Optimizer Step   │
         └────────┬─────────┘
                  │ model (updated)
                  │ optimizer (updated)
                  │
                  └──→ [Back to Prepare Batch with next batch_index]
```

---

## Detailed Connection Map

### Connection 1: Prepare Batch → Forward Pass
```
┌──────────────┐         ┌──────────────┐
│Prepare Batch │         │Forward Pass  │
│              │         │              │
│ Output:      │────────→│ Input:       │
│ batch_data   │         │ batch_data   │
└──────────────┘         └──────────────┘
```

**Action:** Connect `batch_data` output → `batch_data` input

---

### Connection 2: Prepare Batch → Calculate Loss
```
┌──────────────┐         ┌──────────────┐
│Prepare Batch │         │Calculate Loss│
│              │         │              │
│ Output:      │────────→│ Input:       │
│ batch_data   │         │ batch_data   │
└──────────────┘         └──────────────┘
```

**Action:** Connect `batch_data` output → `batch_data` input
**Note:** Same batch_data goes to both Forward Pass AND Calculate Loss

---

### Connection 3: Initialize Optimizer → Zero Gradients
```
┌──────────────────┐    ┌──────────────┐
│Initialize        │    │Zero Gradients│
│Optimizer         │    │              │
│                  │    │              │
│ Output:          │───→│ Input:        │
│ optimizer        │    │ optimizer    │
└──────────────────┘    └──────┬───────┘
                                │ optimizer (zeroed)
                                ↓
```

**Action:** Connect `optimizer` output → `optimizer` input

---

### Connection 4: Model Define → Build Model
```
┌──────────────────┐    ┌──────────────┐
│Model Define (GPU)│    │ Build Model  │
│                  │    │              │
│ Output:          │───→│ Input:       │
│ model_spec       │    │ model_spec   │
└──────────────────┘    └──────┬───────┘
                                │ model
                                ↓
```

**Action:** Connect `model_spec` output → `model_spec` input

### Connection 4b: Build Model → Initialize Optimizer
```
┌──────────────┐    ┌──────────────────┐
│ Build Model  │    │Initialize       │
│              │    │Optimizer        │
│ Output:      │───→│ Input:          │
│ model        │    │ model           │
└──────────────┘    └──────────────────┘
```

**Action:** Connect `model` output → `model` input

### Connection 4c: Build Model → Forward Pass
```
┌──────────────┐    ┌──────────────┐
│ Build Model  │    │Forward Pass  │
│              │    │              │
│ Output:      │───→│ Input:       │
│ model        │    │ model        │
└──────────────┘    └──────────────┘
```

**Action:** Connect `model` output → `model` input (for first forward pass)

---

### Connection 5: Forward Pass → Calculate Loss
```
┌──────────────┐         ┌──────────────┐
│Forward Pass  │         │Calculate Loss│
│              │         │              │
│ Output:      │────────→│ Input:       │
│ predictions  │         │ predictions  │
└──────────────┘         └──────────────┘
```

**Action:** Connect `predictions` output → `predictions` input

---

### Connection 6: Forward Pass → Backward Pass (Model State)
```
┌──────────────┐         ┌──────────────┐
│Forward Pass  │         │Backward Pass │
│              │         │              │
│ Output:      │────────→│ Input:       │
│ model        │         │ model        │
└──────────────┘         └──────────────┘
```

**Action:** Connect `model` output → `model` input

---

### Connection 7: Calculate Loss → Backward Pass
```
┌──────────────┐         ┌──────────────┐
│Calculate Loss│         │Backward Pass │
│              │         │              │
│ Output:      │────────→│ Input:       │
│ loss         │         │ loss         │
└──────────────┘         └──────────────┘
```

**Action:** Connect `loss` output → `loss` input

---

### Connection 8: Zero Gradients → Optimizer Step
```
┌──────────────┐         ┌──────────────┐
│Zero Gradients│         │Optimizer Step│
│              │         │              │
│ Output:      │────────→│ Input:       │
│ optimizer    │         │ optimizer    │
└──────────────┘         └──────────────┘
```

**Action:** Connect `optimizer` output → `optimizer` input

---

### Connection 9: Backward Pass → Optimizer Step
```
┌──────────────┐         ┌──────────────┐
│Backward Pass │         │Optimizer Step│
│              │         │              │
│ Output:      │────────→│ Input:       │
│ model        │         │ model        │
└──────────────┘         └──────────────┘
```

**Action:** Connect `model` output → `model` input

---

## Complete Visual Pipeline

### Full Training Setup

```
┌─────────────────────┐
│   Data Load         │
└──────────┬──────────┘
           │ dataset
           ↓
┌─────────────────────┐
│   Data Split        │
└──────┬──────────────┘
       │ train_dataset
       │ test_dataset
       ↓
┌─────────────────────┐
│ Model Define (GPU)  │
└──────────┬──────────┘
           │ model_spec
           ↓
┌─────────────────────┐
│    Build Model      │
└──────┬──────────────┘
       │ model
       ├─────────────────┐
       │                 │
       ↓                 ↓
┌─────────────────────┐ ┌──────────────┐
│ Initialize Optimizer│ │Forward Pass  │
└──────────┬──────────┘ └──────────────┘
           │ optimizer
           │
           └──→ [Training Loop Starts Below]
```

### Training Loop (One Batch)

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
                    │ optimizer        │ predictions
                    │                  │ model
                    │                  │
                    │                  ↓
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
                    │ Optimizer Step  │
                    └────────┬────────┘
                             │ model (updated)
                             │ optimizer (updated)
                             │
                             └──→ [Next Batch: Prepare Batch (index: 1)]
```

---

## Step-by-Step Connection Instructions

### Setup Connections

1. **Model Define → Initialize Optimizer**
   - Drag edge from Model Define card
   - Connect to Initialize Optimizer card
   - Select: `model_spec` → `model`

### Training Loop Connections

2. **Prepare Batch → Forward Pass**
   - Connect: `batch_data` → `batch_data`

3. **Prepare Batch → Calculate Loss**
   - Connect: `batch_data` → `batch_data`
   - (Same output, two connections)

4. **Initialize Optimizer → Zero Gradients**
   - Connect: `optimizer` → `optimizer`

5. **Forward Pass → Calculate Loss**
   - Connect: `predictions` → `predictions`

6. **Forward Pass → Backward Pass**
   - Connect: `model` → `model`

7. **Calculate Loss → Backward Pass**
   - Connect: `loss` → `loss`

8. **Zero Gradients → Optimizer Step**
   - Connect: `optimizer` → `optimizer`

9. **Backward Pass → Optimizer Step**
   - Connect: `model` → `model`

---

## Key Points

### Multiple Outputs from One Card
- **Prepare Batch** outputs `batch_data` → connect to BOTH Forward Pass AND Calculate Loss
- **Forward Pass** outputs both `predictions` AND `model` → connect separately

### State Flow
- **Model** flows: Forward Pass → Backward Pass → Optimizer Step
- **Optimizer** flows: Zero Gradients → Optimizer Step
- **Batch Data** flows: Prepare Batch → Forward Pass + Calculate Loss

### Loop Structure
- After Optimizer Step, you can connect back to Prepare Batch with the next `batch_index`
- Or use the updated model/optimizer for evaluation

---

## Visual Summary: What Connects to What

```
Prepare Batch
    ├─→ Forward Pass (batch_data)
    └─→ Calculate Loss (batch_data)

Zero Gradients
    └─→ Optimizer Step (optimizer)

Forward Pass
    ├─→ Calculate Loss (predictions)
    └─→ Backward Pass (model)

Calculate Loss
    └─→ Backward Pass (loss)

Backward Pass
    └─→ Optimizer Step (model)

Optimizer Step
    └─→ [Back to Prepare Batch for next iteration]
```

---

## Quick Reference: Connection Checklist

- [ ] Model Define (GPU) → Build Model (`model_spec` → `model_spec`)
- [ ] Build Model → Initialize Optimizer (`model` → `model`)
- [ ] Build Model → Forward Pass (`model` → `model`) [for first iteration]
- [ ] Prepare Batch → Forward Pass (`batch_data` → `batch_data`)
- [ ] Prepare Batch → Calculate Loss (`batch_data` → `batch_data`)
- [ ] Initialize Optimizer → Zero Gradients (`optimizer` → `optimizer`)
- [ ] Forward Pass → Calculate Loss (`predictions` → `predictions`)
- [ ] Forward Pass → Backward Pass (`model` → `model`)
- [ ] Calculate Loss → Backward Pass (`loss` → `loss`)
- [ ] Zero Gradients → Optimizer Step (`optimizer` → `optimizer`)
- [ ] Backward Pass → Optimizer Step (`model` → `model`)
