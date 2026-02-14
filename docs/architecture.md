# Architecture

## Overview

TensorRag is a three-layer system: a React Flow frontend for visual pipeline building, a FastAPI backend for orchestration, and Modal for serverless compute execution.

**Frontend and backend are fully separated** — independent repos, independent runtimes, connected only via REST and WebSocket APIs. This allows each to be developed, deployed, and scaled independently.

```
tensorRag/
├── frontend/          ← Next.js (TypeScript) — runs on :3000
├── backend/           ← FastAPI (Python)     — runs on :8000
└── docs/

┌─────────────────────────────────────────────────┐
│  frontend/  (Next.js + React Flow)              │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │Data Load │──│  Split   │──│  Train   │──►   │
│  └──────────┘  └──────────┘  └──────────┘      │
│                                                 │
│  Exports DAG JSON → sends to backend            │
└────────────────────┬────────────────────────────┘
                     │ REST + WebSocket (http://localhost:8000)
┌────────────────────▼────────────────────────────┐
│  backend/  (FastAPI)                             │
│  - Validates DAG structure                       │
│  - Topological sort → execution order            │
│  - Dispatches each card to Modal                 │
│  - Manages intermediate data via object store    │
│  - Streams execution status via WebSocket        │
│  - Serves card output previews to frontend       │
└────────────────────┬────────────────────────────┘
                     │ Modal SDK
┌────────────────────▼────────────────────────────┐
│  Modal (Serverless Compute)                      │
│  - card_data_load.py                             │
│  - card_split.py                                 │
│  - card_train.py                                 │
│  - card_evaluate.py                              │
│  - card_inference.py                             │
│  Each runs in isolated container w/ optional GPU │
└─────────────────────────────────────────────────┘
```

---

## Layer 1: Frontend

### Tech
- **Next.js** — App router, server components where applicable
- **React Flow** — Canvas for drag-and-drop DAG building
- **Zustand** — Lightweight state management for pipeline state
- **next-themes** — Light/dark mode with system preference detection

### Responsibilities
- **Light / dark mode** — theme toggle in the header, persisted in localStorage, respects system preference on first visit. All components (board, cards, sidebar, config panel, output viewers) adapt to the active theme.
- Render the board (canvas) with a sidebar palette of available cards
- Handle drag-and-drop of cards onto the board
- Manage connections (edges) between cards
- Validate the DAG client-side (no cycles, type-compatible connections)
- Provide per-card configuration UI (click a card → modal/panel with settings)
- Display execution status per card (pending → running → success/fail)
- **Per-card output viewer** — every card has an "Output" tab/button that shows its results after execution:
  - **Data cards** (Load, Split): scrollable data table with column types, row count, shape summary
  - **Train card**: training metrics (loss curve, convergence), model summary (parameters, architecture)
  - **Evaluate card**: metric values (MSE, R2, etc.), comparison charts, confusion matrix (for classification)
  - **Inference card**: predictions table with original features alongside predicted values
- Output viewer renders inline on the card (compact summary) with expand-to-panel for full detail

### Directory Structure
```
frontend/
├── package.json
├── tsconfig.json
├── next.config.ts
├── .env.local              # NEXT_PUBLIC_API_URL=http://localhost:8000
├── tailwind.config.ts      # darkMode: "class" for theme support
├── public/
└── src/
    ├── app/                    # Next.js app router
    ├── components/
    │   ├── board/              # React Flow canvas, controls, minimap
    │   ├── cards/              # Card node components (one per card type)
    │   ├── sidebar/            # Card palette for drag-and-drop
    │   ├── config-panel/       # Card configuration UI
    │   ├── output-viewer/      # Per-card output display components
    │   │   ├── DataTableView   # Tabular data preview (Load, Split, Inference)
    │   │   ├── MetricsView     # Metric values + charts (Evaluate)
    │   │   ├── TrainLogView    # Loss curves, training progress (Train)
    │   │   └── ModelSummary    # Model architecture, parameter count (Train)
    │   └── execution/          # Run button, status indicators, log viewer
    ├── stores/
    │   └── pipeline-store.ts   # Zustand store for pipeline state
    ├── types/
    │   └── pipeline.ts         # Card, Edge, Pipeline type definitions
    └── lib/
        └── api.ts              # Backend API client (REST + WebSocket)
```

### DAG JSON Format
The frontend serializes the pipeline as JSON for the backend:
```json
{
  "pipeline_id": "uuid",
  "nodes": [
    {
      "id": "node-1",
      "type": "data_load",
      "config": {
        "source": "csv",
        "path": "/data/housing.csv"
      },
      "position": { "x": 100, "y": 200 }
    },
    {
      "id": "node-2",
      "type": "split",
      "config": {
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "random_seed": 42
      },
      "position": { "x": 350, "y": 200 }
    }
  ],
  "edges": [
    {
      "source": "node-1",
      "target": "node-2",
      "source_output": "dataset",
      "target_input": "dataset"
    }
  ]
}
```

---

## Layer 2: Backend (Orchestration)

### Tech
- **FastAPI** — REST API + WebSocket support
- **Modal SDK** — To dispatch card computations
- Python 3.11+

### Directory Structure
```
backend/
├── pyproject.toml              # Dependencies, project metadata
├── requirements.txt            # Pinned deps (alt to pyproject.toml)
├── app/
│   ├── main.py                 # FastAPI app, CORS config, router registration
│   ├── config.py               # Settings (Modal token, storage config, etc.)
│   ├── routers/
│   │   ├── pipeline.py         # Pipeline CRUD, validate, execute endpoints
│   │   ├── cards.py            # Card registry, schema endpoints
│   │   └── artifacts.py        # Artifact retrieval, output preview endpoints
│   ├── services/
│   │   ├── dag.py              # DAG validation, topological sort
│   │   ├── executor.py         # Pipeline execution engine
│   │   └── storage.py          # Object store read/write abstraction
│   ├── models/
│   │   ├── pipeline.py         # Pydantic models for pipeline, node, edge
│   │   └── card.py             # Card interface, schemas
│   └── ws/
│       └── status.py           # WebSocket manager for live updates
├── cards/
│   ├── base.py                 # CardInterface base class
│   ├── data_load.py            # Modal function: load data
│   ├── data_split.py           # Modal function: train/test split
│   ├── model_define.py         # Model specification (no Modal execution)
│   ├── train.py                # Modal function: training loop
│   ├── evaluate.py             # Modal function: metrics + charts
│   └── inference.py            # Modal function: run predictions
└── tests/
    ├── test_dag.py
    ├── test_executor.py
    └── test_cards.py
```

### Responsibilities
- Receive pipeline DAG JSON from the frontend
- Validate DAG structure (acyclic, valid connections, required configs present)
- Perform topological sort to determine execution order
- Execute cards sequentially or in parallel based on dependencies
- Manage data transport between cards (read/write to object store)
- Stream execution status updates to frontend via WebSocket
- Handle errors, retries, and partial pipeline re-runs
- CORS configured to allow requests from frontend origin

### API Endpoints
```
POST   /api/pipeline/validate     — Validate a DAG before execution
POST   /api/pipeline/execute      — Start pipeline execution
GET    /api/pipeline/{id}/status  — Get execution status
WS     /ws/pipeline/{id}          — WebSocket for live status updates
GET    /api/cards                 — List available card types and schemas
GET    /api/artifacts/{id}        — Retrieve intermediate/final results
GET    /api/card/{id}/output      — Get a card's output for the output viewer
```

### Execution Engine — Hybrid (Local + Modal)

Not every card needs a remote container. The executor checks each card's `execution_mode` and routes accordingly:

| Card | execution_mode | Why |
|------|---------------|-----|
| Data Load | `local` | Reading a CSV/URL — no heavy compute, instant on backend |
| Data Split | `local` | Pandas operation, milliseconds |
| Model Define | `local` | Config-only, no computation at all |
| Train | `modal` | May need GPU, long-running, resource-intensive |
| Evaluate | `local` | Metrics computation on small test set, lightweight |
| Inference | `modal` | May need GPU for large models, batch predictions |

**Rules:**
- `local` cards run directly in the FastAPI process (or a background thread)
- `modal` cards are dispatched to Modal's serverless containers
- A card's `execution_mode` is part of its interface — the executor doesn't guess
- Future DL cards will always be `modal` (GPU required)

```python
# Pseudocode for hybrid DAG execution
def execute_pipeline(dag):
    execution_order = topological_sort(dag)

    for level in execution_order:
        tasks = []
        for card in level:
            input_data = gather_inputs(card, object_store)

            if card.execution_mode == "local":
                task = run_local(card.type, card.config, input_data)
            else:
                task = dispatch_to_modal(card.type, card.config, input_data)

            tasks.append(task)

        results = await asyncio.gather(*tasks)

        for card, result in zip(level, results):
            store_output(card, result, object_store)
            notify_frontend(card.id, status="completed")
```

---

## Layer 3: Compute (Modal)

### Tech
- **Modal** — Serverless Python functions with optional GPU
- Only cards with `execution_mode: "modal"` run here

### When Modal is Used
- **Training** — model fitting, especially when GPU is needed (DL cards in future)
- **Inference** — batch predictions on large datasets or GPU-bound models
- Any future card that requires heavy compute, special hardware, or large memory

### When Modal is NOT Used
- **Data Load** — file I/O, pandas read — runs locally on the backend
- **Data Split** — in-memory array slicing — runs locally
- **Model Define** — pure config, no execution — runs locally
- **Evaluate** — metrics on small test sets — runs locally

### Card Function Structure (Modal cards only)
```python
@modal.function(image=ml_image, gpu="T4")
def card_train(config: dict, input_ref: str) -> str:
    # 1. Load input from object store
    data = load_from_store(input_ref)

    # 2. Execute card logic
    model = build_model(config)
    model.fit(data["X_train"], data["y_train"])

    # 3. Save output to object store
    output_ref = save_to_store({"model": model, "history": history})
    return output_ref
```

### Local Card Structure (backend-local cards)
```python
# Runs directly in the FastAPI process — no container overhead
def card_data_load(config: dict, inputs: dict) -> dict:
    if config["source"] == "csv":
        df = pd.read_csv(config["path"])
    elif config["source"] == "url":
        df = pd.read_csv(config["url"])

    output_ref = save_to_store({"dataset": df})
    return {"dataset": output_ref}
```

---

## Data Transport

This is the critical connective layer. Each card reads its input from and writes its output to an object store. The backend coordinates which references to pass between cards.

### Flow
```
Card A executes → writes output to store → returns reference (key/path)
                                                    │
Backend receives reference, maps it to Card B's input
                                                    │
Card B executes → reads input from store using reference → processes → writes output
```

### Options Under Consideration
| Option | Pros | Cons |
|--------|------|------|
| Modal Volumes | Native to Modal, fast | Vendor lock-in |
| S3/GCS | Standard, portable | Extra infra, latency |
| Modal Dict/Queue | Simple for small data | Doesn't scale |

---

## Card Interface Contract

Every card must conform to this interface:

```python
class CardInterface:
    """Standard contract for all pipeline cards."""

    # Metadata
    card_type: str          # e.g., "data_load", "train"
    display_name: str       # e.g., "Data Loader"
    description: str
    category: str           # e.g., "data", "model", "evaluation"

    # Execution routing
    execution_mode: str     # "local" or "modal"
    gpu: str | None         # None for local/CPU, "T4"/"A100" for Modal GPU cards

    # Schema
    config_schema: dict     # JSON Schema for card configuration
    input_schema: dict      # Expected input types/shapes
    output_schema: dict     # Output types/shapes

    # Output viewer
    output_view_type: str   # "table", "metrics", "model_summary", "chart"

    # Execution
    def execute(config: dict, inputs: dict) -> dict:
        """Run the card's logic. Returns outputs matching output_schema."""
        ...

    def get_output_preview(output: dict) -> dict:
        """Return a frontend-friendly preview of the output for the card viewer.
        Includes summary stats, truncated data, or chart-ready payloads."""
        ...
```

### Type Compatibility
Connections between cards are validated:
- Card A's `output_schema` must be compatible with Card B's `input_schema`
- The frontend enforces this when users draw edges
- The backend double-checks before execution

---

## Frontend ↔ Backend Separation

The frontend and backend are **completely independent**:

| Aspect | Frontend | Backend |
|--------|----------|---------|
| Language | TypeScript | Python |
| Runtime | Node.js / Browser | Python 3.11+ / Uvicorn |
| Package Manager | npm / pnpm | pip / uv |
| Deploy independently | Yes (Vercel, Netlify) | Yes (Railway, Fly.io, any VPS) |
| Communication | HTTP REST + WebSocket | Serves REST + WebSocket |
| Shared code | None | None |

### Why Separate?
- **Independent deployment** — frontend can be on a CDN, backend on a GPU-capable server
- **Team scalability** — frontend and backend devs work in isolated codebases
- **Tech freedom** — no compromise on tooling (TypeScript vs Python)
- **Replaceability** — either side can be swapped without touching the other (e.g., swap Next.js for a mobile app)

### Connection Config
- Frontend reads `NEXT_PUBLIC_API_URL` from `.env.local` (defaults to `http://localhost:8000`)
- Backend configures CORS to allow the frontend origin
- WebSocket connects to `ws://localhost:8000/ws/pipeline/{id}`

---

## Security Considerations

- User-uploaded data is sandboxed within Modal containers
- No arbitrary code execution — users configure cards, not write code
- API authentication (TBD — JWT or API keys)
- Rate limiting on pipeline execution
- CORS restricted to known frontend origins
