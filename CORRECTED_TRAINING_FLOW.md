# Corrected Training Flow - Single Connection Per Output

## Important Constraint

Each card output can only connect to **one input**. This means we need to structure the flow differently.

---

## Corrected Model Setup Flow

### Option 1: Sequential Flow (Recommended)

```
Model Define (GPU)
    ↓ model_spec
Build Model
    ↓ model
Initialize Optimizer
    ├─→ optimizer (output 1)
    └─→ model (output 2)
```

**Connections:**
1. Model Define (GPU) `model_spec` → Build Model `model_spec`
2. Build Model `model` → Initialize Optimizer `model`
3. Initialize Optimizer `optimizer` → Zero Gradients `optimizer`
4. Initialize Optimizer `model` → Forward Pass `model` (first batch)

**Note:** Initialize Optimizer now outputs **both** `optimizer` and `model`, allowing you to connect them separately.

---

## Complete Corrected Flow

### Setup Phase

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
└──────┬──────────────┘
       │
       ├─→ optimizer ──→ Zero Gradients
       │
       └─→ model ──→ Forward Pass (first batch)
```

### Training Loop (One Batch)

```
Prepare Batch
    ├─→ batch_data ──→ Forward Pass
    │
    └─→ batch_data ──→ Calculate Loss

Zero Gradients
    └─→ optimizer ──→ Optimizer Step

Forward Pass
    ├─→ predictions ──→ Calculate Loss
    │
    └─→ model ──→ Backward Pass

Calculate Loss
    └─→ loss ──→ Backward Pass

Backward Pass
    └─→ model ──→ Optimizer Step

Optimizer Step
    ├─→ model ──→ Forward Pass (next batch)
    │
    └─→ optimizer ──→ Zero Gradients (next batch)
```

---

## Key Points

1. **Initialize Optimizer** outputs TWO things:
   - `optimizer` → connects to Zero Gradients
   - `model` → connects to Forward Pass

2. **Prepare Batch** outputs `batch_data` that connects to BOTH:
   - Forward Pass (needs X)
   - Calculate Loss (needs y)

3. **Forward Pass** outputs TWO things:
   - `predictions` → connects to Calculate Loss
   - `model` → connects to Backward Pass

4. **Optimizer Step** outputs TWO things:
   - `model` → connects to Forward Pass (next batch)
   - `optimizer` → connects to Zero Gradients (next batch)

---

## Visual Connection Map

```
SETUP:
Model Define (GPU) → Build Model → Initialize Optimizer
                                          ├─→ optimizer → Zero Gradients
                                          └─→ model → Forward Pass

TRAINING LOOP (per batch):
Prepare Batch ──┬─→ Forward Pass ──┬─→ Calculate Loss ──→ Backward Pass ──→ Optimizer Step
                └─→ Calculate Loss ─┘                                    │
                                                                         │
Zero Gradients ────────────────────────────────────────────────────────┘
                                                                         │
                                                                         ├─→ model → Forward Pass (next)
                                                                         └─→ optimizer → Zero Gradients (next)
```

---

## Step-by-Step Connection Instructions

### Setup Connections

1. **Model Define (GPU)** → **Build Model**
   - Connect: `model_spec` → `model_spec`

2. **Build Model** → **Initialize Optimizer**
   - Connect: `model` → `model`

3. **Initialize Optimizer** → **Zero Gradients**
   - Connect: `optimizer` → `optimizer`

4. **Initialize Optimizer** → **Forward Pass** (first batch)
   - Connect: `model` → `model`

### Training Loop Connections (per batch)

5. **Prepare Batch** → **Forward Pass**
   - Connect: `batch_data` → `batch_data`

6. **Prepare Batch** → **Calculate Loss**
   - Connect: `batch_data` → `batch_data`
   - **Note:** You'll need to create TWO separate edges from Prepare Batch

7. **Forward Pass** → **Calculate Loss**
   - Connect: `predictions` → `predictions`

8. **Forward Pass** → **Backward Pass**
   - Connect: `model` → `model`

9. **Calculate Loss** → **Backward Pass**
   - Connect: `loss` → `loss`

10. **Backward Pass** → **Optimizer Step**
    - Connect: `model` → `model`

11. **Optimizer Step** → **Forward Pass** (next batch)
    - Connect: `model` → `model`

12. **Optimizer Step** → **Zero Gradients** (next batch)
    - Connect: `optimizer` → `optimizer`

---

## Important Notes

- **Multiple outputs are supported** - a card can have multiple output handles
- **Each output handle can connect to ONE input** - you can't split one output to multiple inputs
- **Solution:** Cards with multiple outputs (like Initialize Optimizer, Forward Pass, Optimizer Step) provide separate handles for each output
- **Prepare Batch** needs to connect `batch_data` to both Forward Pass and Calculate Loss - create two separate edges

---

## Updated Card Outputs

### Initialize Optimizer
- Outputs: `optimizer`, `model`

### Forward Pass
- Outputs: `predictions`, `model`

### Optimizer Step
- Outputs: `model`, `optimizer`

### Prepare Batch
- Outputs: `batch_data`, `batch_info`
- **Note:** `batch_data` needs to connect to TWO places (Forward Pass and Calculate Loss)
