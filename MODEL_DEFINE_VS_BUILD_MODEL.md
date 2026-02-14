# Model Define vs Build Model - Explained

## Overview

These are **two separate steps** in the model creation process. Here's why we need both:

---

## Model Define (GPU)

### What It Does
- **Creates a specification** (JSON config) describing the model architecture
- **No actual model is created** - just a configuration file
- Runs **locally** (execution_mode: "local")
- **Lightweight** - just saves a JSON file

### Output
- `model_spec` (JSON) - Contains:
  - Model architecture (hidden layers, activation, dropout)
  - Training hyperparameters (learning rate, epochs, batch size)

### Example Output
```json
{
  "model_type": "deep_neural_network",
  "hidden_layers": [128, 64, 32],
  "activation": "relu",
  "dropout": 0.2,
  "learning_rate": 0.001,
  "epochs": 50,
  "batch_size": 32
}
```

### When to Use
- First step in model setup
- Define your architecture before building
- Can be reused with different input sizes

---

## Build Model

### What It Does
- **Creates an actual PyTorch model object** from the specification
- **Instantiates the neural network** with real layers
- Runs on **Modal** (execution_mode: "modal") - can use GPU
- **Heavy operation** - creates the actual model

### Input
- `model_spec` (JSON) - from Model Define (GPU)
- `input_size` (config) - number of input features

### Output
- `model` (PyTorch model object) - Actual neural network ready for training

### What Happens Inside
```python
# Takes model_spec JSON
spec = {
    "hidden_layers": [128, 64, 32],
    "activation": "relu",
    "dropout": 0.2
}

# Creates actual PyTorch model
model = NeuralNetwork(
    input_size=13,  # From config
    hidden_layers=[128, 64, 32],  # From spec
    activation="relu",  # From spec
    dropout=0.2  # From spec
)
```

### When to Use
- After Model Define (GPU)
- When you need the actual model object
- Before Initialize Optimizer or Forward Pass

---

## Why Two Separate Cards?

### Separation of Concerns

1. **Model Define** = **Design Phase**
   - Think of it as a blueprint
   - Defines WHAT the model should look like
   - Can be saved, shared, versioned
   - No computation needed

2. **Build Model** = **Construction Phase**
   - Think of it as building from the blueprint
   - Creates the ACTUAL model
   - Requires computation (GPU/CPU)
   - Needs specific input size

### Benefits

1. **Flexibility**
   - Same model spec can be built with different input sizes
   - Can experiment with architectures without rebuilding

2. **Efficiency**
   - Model Define runs locally (fast)
   - Build Model runs on Modal (can use GPU if needed)

3. **Reusability**
   - Model spec can be reused
   - Can build multiple models from one spec

4. **Clarity**
   - Clear separation: design vs construction
   - Easier to understand the pipeline

---

## Visual Comparison

### Model Define (GPU)
```
Input:  None (just configuration)
        ↓
Process: Creates JSON specification
        ↓
Output: model_spec (JSON file)
        {
          "hidden_layers": [128, 64, 32],
          "activation": "relu",
          ...
        }
```

### Build Model
```
Input:  model_spec (JSON) + input_size (config)
        ↓
Process: Creates PyTorch NeuralNetwork object
        ↓
Output: model (PyTorch model object)
        NeuralNetwork(
          input_size=13,
          hidden_layers=[128, 64, 32],
          ...
        )
```

---

## Complete Flow

```
┌─────────────────────┐
│ Model Define (GPU)  │  ← Design the architecture
│                     │     (JSON specification)
│ Config:             │
│ - hidden_layers     │
│ - activation        │
│ - dropout           │
│ - learning_rate     │
│ - epochs            │
│ - batch_size        │
└──────────┬──────────┘
           │ model_spec (JSON)
           ↓
┌─────────────────────┐
│    Build Model       │  ← Create actual model
│                     │     (PyTorch object)
│ Config:             │
│ - input_size: 13    │
│                     │
│ Uses model_spec to  │
│ build NeuralNetwork │
└──────────┬──────────┘
           │ model (PyTorch object)
           ↓
    [Ready for training]
```

---

## Analogy

Think of it like building a house:

- **Model Define (GPU)** = **Architect's Blueprint**
  - Draws the plans
  - Specifies dimensions, materials, layout
  - Just a document/plan

- **Build Model** = **Construction**
  - Actually builds the house from the blueprint
  - Creates the real structure
  - Requires materials and labor

---

## When You Might Skip Build Model

If you use the **full Train (GPU) card**, it handles both:
- Takes `model_spec` directly
- Builds the model internally
- Trains it

So the flow becomes:
```
Model Define (GPU) → Train (GPU)
```

But with **individual training steps**, you need Build Model because:
- Initialize Optimizer needs an actual model object
- Forward Pass needs an actual model object
- They can't work with just a JSON spec

---

## Summary

| Aspect | Model Define (GPU) | Build Model |
|--------|-------------------|-------------|
| **Purpose** | Create specification | Create actual model |
| **Output Type** | JSON (text) | PyTorch model (object) |
| **Execution** | Local (fast) | Modal (GPU-capable) |
| **Input** | Configuration only | model_spec + input_size |
| **Output** | model_spec | model |
| **Reusable** | Yes (can build multiple models) | No (specific instance) |
| **When to Use** | First step | After Model Define |

---

## Recommendation

**Always use both in sequence:**
1. Model Define (GPU) - Define your architecture
2. Build Model - Create the actual model

This gives you maximum flexibility and clarity in your pipeline!
