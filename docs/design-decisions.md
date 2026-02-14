# Design Decisions

Open questions and trade-offs that need resolution before implementation.

---

## Decision 1: Data Transport Between Cards

**Status: OPEN**

How does data flow from one card's output to the next card's input?

### Option A: Modal Volumes (Recommended for V1)
Each card writes output (dataframes, model weights, metrics) to a Modal Volume. The backend passes volume paths between cards.

- **Pros:** Native to Modal, no extra infra, fast reads/writes within Modal
- **Cons:** Vendor lock-in to Modal, harder to inspect data locally
- **Best for:** Getting to MVP quickly

### Option B: S3-Compatible Object Store
Use MinIO (local dev) or S3 (production). Each card uploads/downloads via presigned URLs.

- **Pros:** Portable, standard, works with any compute backend
- **Cons:** Extra service to manage, network latency between Modal and S3
- **Best for:** Long-term production architecture

### Option C: Hybrid
Use Modal Volumes for inter-card data during execution, but persist final artifacts (trained models, evaluation reports) to S3 for durability.

- **Pros:** Fast execution + durable outputs
- **Cons:** Two storage systems to manage

**Recommendation:** Start with Option A (Modal Volumes) for V1, migrate to Option C when needed.

---

## Decision 2: Card Granularity

**Status: OPEN**

How atomic should each card be?

### Coarse-Grained
Few cards, each does a lot:
- `Data Load` → `Train` → `Evaluate`
- Simpler UX, fewer connections to manage
- Less flexible, harder to reuse parts

### Fine-Grained
Many small cards:
- `Load CSV` → `Clean` → `Feature Engineer` → `Split` → `Define Model` → `Set Optimizer` → `Train Loop` → `Evaluate` → `Export`
- Maximum flexibility and reuse
- More complex UX, more connections

### Recommended: Medium Granularity for V1
Start with these cards:
1. **Data Load** — Load data from CSV, Parquet, or URL
2. **Data Split** — Train/test/validation split
3. **Model Define** — Choose architecture and hyperparameters
4. **Train** — Training loop with configurable epochs, optimizer, loss
5. **Evaluate** — Metrics computation on test set
6. **Inference** — Run predictions on new data

This gives enough granularity to be useful without overwhelming the UX. More atomic cards can be added later.

---

## Decision 3: Frontend State Management

**Status: OPEN**

### Option A: Zustand (Recommended)
- Lightweight, minimal boilerplate
- Works well with React Flow (both use immutable updates)
- Simple to learn and debug

### Option B: Redux Toolkit
- Battle-tested, great devtools
- More boilerplate than Zustand
- Overkill for V1

### Option C: React Flow's Built-in State
- React Flow manages nodes/edges internally
- Works for simple cases but limited for pipeline metadata, execution state, etc.

**Recommendation:** Zustand for pipeline state + React Flow's internal state for canvas interactions.

---

## Decision 4: Backend Framework

**Status: OPEN**

### Option A: FastAPI (Recommended)
- Python-native — same language as ML code
- Built-in WebSocket support for live status
- Async by default, good for dispatching Modal tasks
- OpenAPI docs auto-generated

### Option B: Next.js API Routes
- Single deployment (frontend + backend together)
- But ML orchestration in JavaScript is unnatural
- Would still need Python for Modal functions

### Option C: FastAPI + Next.js API Routes (Hybrid)
- Next.js handles auth, session, UI-facing APIs
- FastAPI handles pipeline execution, Modal dispatch
- More complex but cleaner separation

**Recommendation:** FastAPI as the sole backend for V1. Add Next.js API routes only if auth/session management justifies it later.

---

## Decision 5: Card Configuration UI

**Status: OPEN**

When a user clicks a card on the board, how do they configure it?

### Option A: Side Panel
- Panel slides in from the right (like Figma's property panel)
- Card stays visible on the canvas
- Good for quick edits

### Option B: Modal Dialog
- Full modal overlay with form fields
- More space for complex configurations
- Hides the canvas temporarily

### Option C: Inline on Card
- Configuration directly on the card node
- Limited space, only works for simple configs

**Recommendation:** Side panel (Option A) as default, with an "expand" button to open a full modal for cards with complex configuration (like model architecture).

---

## Decision 6: Execution Model

**Status: OPEN**

### Synchronous (Simple)
- User clicks "Run"
- Backend executes cards one by one (respecting DAG order)
- Frontend polls or uses WebSocket for updates
- Pipeline must complete in one session

### Asynchronous with Job Queue
- User clicks "Run"
- Backend creates a job, returns immediately
- Worker processes execute cards asynchronously
- User can close browser and come back to results
- More complex but production-ready

**Recommendation:** Synchronous for V1 with WebSocket status streaming. Move to async job queue when pipelines get long-running.

---

## Decision 7b: Execution Routing (Local vs Modal)

**Status: DECIDED — Hybrid**

Not every card needs a remote Modal container. Cards that don't need GPU or heavy compute run directly on the backend.

### Per-Card Routing

| Card | Execution | Rationale |
|------|-----------|-----------|
| Data Load | **Local** | File I/O, pandas read — no reason to spin up a container |
| Data Split | **Local** | In-memory array slicing, milliseconds |
| Model Define | **Local** | Pure config, zero computation |
| Train | **Modal** | Model fitting, potentially GPU-bound, long-running |
| Evaluate | **Local** | Metrics on a small test set, lightweight |
| Inference | **Modal** | Batch predictions, may need GPU for large models |

### How It Works
- Each card declares `execution_mode: "local" | "modal"` in its interface
- The executor checks this field and routes accordingly
- Local cards run in the FastAPI process (or a background thread)
- Modal cards are dispatched via the Modal SDK
- Both types follow the same `CardInterface` contract

### Why Not Just Run Everything on Modal?
- **Latency:** Modal cold starts add 2-10s per card. Data Load + Split + Model Define + Evaluate = 4 unnecessary cold starts
- **Cost:** Modal charges per compute-second. Loading a CSV doesn't need a container
- **Simplicity:** Local execution has zero infrastructure dependencies for lightweight ops
- **Dev experience:** Local cards are instantly testable without a Modal account

---

## Decision 7: Serialization Format for Intermediate Data

**Status: OPEN**

How are dataframes, models, and other objects serialized between cards?

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| Pickle | Python objects, models | Fast, preserves types | Security risk, Python-only |
| Parquet | Tabular data | Fast, columnar, language-agnostic | Only for tabular data |
| JSON | Configs, metrics | Universal, human-readable | Slow for large data |
| Joblib | sklearn models | Optimized for numpy arrays | Python-only |
| SafeTensors | PyTorch/TF models | Safe, fast | ML frameworks only |

**Recommendation:** Use Parquet for tabular data, SafeTensors/Joblib for models, JSON for configs and metrics. Each card declares its output format in its schema.

---

## Decision 8: Theming (Light / Dark Mode)

**Status: DECIDED — next-themes + Tailwind `dark:` classes**

The dashboard supports light and dark mode.

### Approach
- **next-themes** handles theme switching, localStorage persistence, and system preference detection
- **Tailwind CSS** `darkMode: "class"` — all components use `dark:` variant classes
- Theme toggle in the dashboard header (light / dark / system)
- React Flow canvas, cards, sidebar, config panel, output viewers all respect the active theme

### Why This Approach?
- next-themes is the standard for Next.js — handles SSR flash-of-wrong-theme, localStorage, and system media query
- Tailwind `dark:` classes keep styling co-located with components, no separate theme files
- No runtime CSS-in-JS overhead

### Design Tokens (CSS Variables)
All colors go through CSS variables so cards, charts, and output viewers stay consistent:
```css
:root {
  --bg-primary: #ffffff;
  --bg-card: #f8f9fa;
  --text-primary: #1a1a1a;
  --border: #e2e8f0;
  --accent: #3b82f6;
}

.dark {
  --bg-primary: #0f172a;
  --bg-card: #1e293b;
  --text-primary: #f1f5f9;
  --border: #334155;
  --accent: #60a5fa;
}
```

---

## Decisions Summary

| # | Decision | Status | Current Leaning |
|---|----------|--------|-----------------|
| 1 | Data Transport | OPEN | Modal Volumes for V1 |
| 2 | Card Granularity | OPEN | Medium (6 cards for V1) |
| 3 | State Management | OPEN | Zustand |
| 4 | Backend Framework | OPEN | FastAPI |
| 5 | Card Config UI | OPEN | Side panel + expand to modal |
| 6 | Execution Model | OPEN | Synchronous + WebSocket |
| 7a | Serialization | OPEN | Parquet + SafeTensors + JSON |
| 7b | Execution Routing | DECIDED | Hybrid — local for lightweight, Modal for heavy compute |
| 8 | Theming | DECIDED | next-themes + Tailwind `dark:` classes |
