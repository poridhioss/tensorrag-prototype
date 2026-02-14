# TensorRag Architecture Overview

## System Overview

TensorRag is a **visual ML pipeline builder** that allows users to create machine learning workflows by dragging and connecting cards on a canvas. The system consists of three main layers:

1. **Frontend** (Next.js + React Flow) - Visual interface for building pipelines
2. **Backend** (FastAPI) - Orchestration and execution engine
3. **Compute** (Modal) - Serverless execution for heavy compute tasks

---

## Frontend Architecture

### Technology Stack
- **Next.js 16** with App Router
- **React Flow** for drag-and-drop DAG visualization
- **Zustand** for state management
- **TypeScript** for type safety
- **Tailwind CSS v3** for styling (with dark mode support)

### Key Components

#### 1. **State Management** (`src/store/pipelineStore.ts`)
- Manages pipeline state: nodes, edges, card schemas
- Handles node status (pending/running/completed/failed)
- Stores node configurations and output previews
- Provides methods to convert pipeline to API request format

**Key State:**
- `nodes`: Array of React Flow nodes (cards on canvas)
- `edges`: Array of connections between nodes
- `cardSchemas`: Available card types from backend
- `pipelineId`: Current pipeline identifier
- `isExecuting`: Execution status flag

#### 2. **API Client** (`src/lib/api.ts`)
- REST API wrapper for backend communication
- Endpoints:
  - `fetchCards()` - Get available card types
  - `validatePipeline()` - Validate DAG structure
  - `executePipeline()` - Start pipeline execution
  - `fetchCardOutput()` - Get card output preview
  - `getArtifactUrl()` - Get artifact download URL

#### 3. **Pipeline Runner** (`src/lib/pipelineRunner.ts`)
- Orchestrates pipeline execution flow
- Validates pipeline before execution
- Connects WebSocket for real-time updates
- Handles node status updates and output fetching

#### 4. **WebSocket Hook** (`src/hooks/useWebSocket.ts`)
- Manages WebSocket connection to backend
- Receives real-time status updates:
  - `node_status` - Node execution status changes
  - `log` - Execution log messages

#### 5. **UI Components**

**PipelineCanvas** (`src/components/canvas/PipelineCanvas.tsx`)
- Main React Flow canvas
- Handles drag-and-drop of cards
- Manages node selection and connections
- Includes minimap and controls

**CardNode** (`src/components/canvas/CardNode.tsx`)
- Custom React Flow node component
- Displays card status, name, and output preview
- Shows error states and execution indicators

**CardPalette** (`src/components/sidebar/CardPalette.tsx`)
- Sidebar with draggable card types
- Organized by category (data, model, evaluation, inference)

**ConfigPanel** (`src/components/config/ConfigPanel.tsx`)
- Configuration form for selected card
- Uses JSON Schema to render form fields
- Updates node configuration in store

**OutputViewer** (`src/components/output/OutputViewer.tsx`)
- Displays card output previews
- Supports different view types:
  - `table` - Data tables (DataFrame preview)
  - `metrics` - Training/evaluation metrics
  - `model_summary` - Model architecture and parameters

### Data Flow

```
User Action (drag card) 
  → Add to store.nodes
  → Render on canvas

User Action (connect cards)
  → Validate type compatibility
  → Add to store.edges

User Action (run pipeline)
  → Validate pipeline
  → Connect WebSocket
  → POST /api/pipeline/execute
  → Receive WebSocket updates
  → Update node statuses
  → Fetch output previews
```

---

## Backend Architecture

### Technology Stack
- **FastAPI** - REST API + WebSocket server
- **Python 3.11+**
- **Modal SDK** - Serverless function dispatch
- **Pydantic** - Data validation and models

### Directory Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app, CORS, routes
│   ├── config.py            # Settings (storage, Modal, S3)
│   ├── models/
│   │   ├── pipeline.py     # Pydantic models (PipelineRequest, NodeStatus)
│   │   └── card.py         # CardSchema model
│   ├── routers/
│   │   ├── pipeline.py     # Pipeline endpoints
│   │   ├── cards.py        # Card registry endpoint
│   │   └── artifacts.py    # Artifact retrieval
│   ├── services/
│   │   ├── dag.py          # DAG validation, topological sort
│   │   ├── executor.py     # Pipeline execution engine
│   │   ├── storage.py      # Local filesystem storage
│   │   └── s3_storage.py   # S3 storage (optional)
│   └── ws/
│       └── status.py       # WebSocket manager
└── cards/
    ├── base.py             # BaseCard abstract class
    ├── registry.py         # Card registry
    ├── modal_app.py        # Modal app definition
    ├── data_load.py        # Data loading card
    ├── data_split.py       # Train/test split card
    ├── model_define.py     # Model specification card
    ├── train.py            # Training card
    ├── evaluate.py         # Evaluation card
    └── inference.py        # Inference card
```

### Key Components

#### 1. **Pipeline Router** (`app/routers/pipeline.py`)
- `POST /api/pipeline/validate` - Validate DAG structure
- `POST /api/pipeline/execute` - Start pipeline execution (async)
- `GET /api/pipeline/{id}/status` - Get execution status

#### 2. **Cards Router** (`app/routers/cards.py`)
- `GET /api/cards` - List all available card types with schemas

#### 3. **Executor Service** (`app/services/executor.py`)
- **Core execution engine**
- Performs topological sort to determine execution order
- Routes cards to local or Modal execution
- Manages data flow between cards via storage service
- Sends WebSocket updates for status changes

**Execution Flow:**
```
1. Validate DAG (no cycles, valid connections)
2. Topological sort → execution levels
3. For each level:
   - Gather inputs from upstream nodes
   - Execute card (local or Modal)
   - Save outputs to storage
   - Update status via WebSocket
4. Return final state
```

#### 4. **Storage Service** (`app/services/storage.py`)
- **Abstraction for data persistence**
- Methods:
  - `save_dataframe()` / `load_dataframe()` - Parquet files
  - `save_model()` / `load_model()` - Joblib files
  - `save_json()` / `load_json()` - JSON files
  - `save_bytes()` / `load_bytes()` - Binary files
- Storage path: `{STORAGE_DIR}/{pipeline_id}/{node_id}/{key}.{ext}`

#### 5. **DAG Service** (`app/services/dag.py`)
- `validate_dag()` - Check for cycles, valid connections
- `topological_sort()` - Determine execution order

#### 6. **WebSocket Manager** (`app/ws/status.py`)
- Manages WebSocket connections per pipeline
- Broadcasts status updates:
  - `node_status` - Node execution status
  - `log` - Execution log messages

#### 7. **Card System**

**BaseCard** (`cards/base.py`)
- Abstract base class for all cards
- Defines interface:
  - `execute()` - Run card logic
  - `get_output_preview()` - Generate frontend preview
  - `to_schema()` - Convert to API schema

**Card Registry** (`cards/registry.py`)
- Registers all card implementations
- Provides `get_card()` and `list_cards()` functions

**Card Implementations:**
- **DataLoadCard** - Load CSV, URL, or sample datasets
- **DataSplitCard** - Split dataset into train/test
- **ModelDefineCard** - Define model specification
- **TrainCard** - Train model on training data
- **EvaluateCard** - Evaluate model on test data
- **InferenceCard** - Run predictions on new data

### Execution Modes

Cards can run in two modes:

1. **Local** (`execution_mode: "local"`)
   - Runs directly in FastAPI process
   - Used for: data loading, splitting, model definition, evaluation
   - Fast, no container overhead

2. **Modal** (`execution_mode: "modal"`)
   - Dispatched to Modal serverless containers
   - Used for: training, inference (GPU-capable)
   - Serializes inputs → Modal → deserializes outputs

---

## Modal Integration

### Modal App (`cards/modal_app.py`)
- Defines Modal app: `tensorrag`
- Single generic function: `run_card()`
- Accepts: `card_type`, `config`, `serialized_inputs`
- Returns: `serialized_outputs`

### Execution Flow (Modal)
```
Backend:
  1. Serialize inputs (DataFrame → Parquet bytes, Model → Joblib bytes)
  2. Call modal.Function.from_name("tensorrag", "run_card")
  3. Dispatch with serialized data

Modal Container:
  1. Deserialize inputs
  2. Get card from registry
  3. Execute card logic
  4. Serialize outputs
  5. Return to backend

Backend:
  1. Deserialize outputs
  2. Save to local storage
  3. Update pipeline state
```

---

## Data Flow Between Cards

### Storage Reference System

Each card execution:
1. **Reads inputs** from storage using references (file paths)
2. **Processes data** (e.g., train model, split data)
3. **Saves outputs** to storage, returns references
4. **References passed** to downstream cards via edges

### Example Flow:
```
DataLoadCard
  → saves "dataset" → returns {"dataset": "/storage/p1/n1/dataset.parquet"}
  → reference passed to DataSplitCard via edge

DataSplitCard
  → reads "/storage/p1/n1/dataset.parquet"
  → splits into train/test
  → saves "train_dataset" and "test_dataset"
  → returns {"train_dataset": "/storage/p1/n2/train_dataset.parquet", ...}
```

---

## API Contract

### Request/Response Types

**PipelineRequest:**
```typescript
{
  pipeline_id: string;
  nodes: Array<{
    id: string;
    type: string;  // card_type
    config: Record<string, unknown>;
    position: { x: number; y: number };
  }>;
  edges: Array<{
    source: string;  // node id
    target: string;  // node id
    source_output: string;  // output key
    target_input: string;   // input key
  }>;
}
```

**CardSchema:**
```typescript
{
  card_type: string;
  display_name: string;
  description: string;
  category: "data" | "model" | "evaluation" | "inference";
  execution_mode: "local" | "modal";
  config_schema: JsonSchema;  // For form generation
  input_schema: Record<string, string>;  // {input_name: type}
  output_schema: Record<string, string>;  // {output_key: type}
  output_view_type: "table" | "metrics" | "model_summary";
}
```

**WebSocket Messages:**
```typescript
// Node status update
{
  type: "node_status";
  node_id: string;
  status: "pending" | "running" | "completed" | "failed";
  message: string;
  timestamp: string;
}

// Log message
{
  type: "log";
  text: string;
  timestamp: string;
}
```

---

## Configuration

### Frontend (`.env.local`)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### Backend (`.env`)
```env
FRONTEND_ORIGIN=http://localhost:3000
STORAGE_DIR=./storage
MODAL_ENABLED=true
S3_ENABLED=false
```

---

## Key Design Decisions

1. **Separation of Concerns**
   - Frontend and backend are completely independent
   - No shared code, only API contracts
   - Can be deployed separately

2. **Card-Based Architecture**
   - Each card is self-contained
   - Standard interface (BaseCard)
   - Easy to add new cards

3. **Hybrid Execution**
   - Lightweight cards run locally
   - Heavy compute runs on Modal
   - Transparent to user

4. **Storage Abstraction**
   - Cards don't know about storage implementation
   - Can switch between local filesystem and S3
   - References passed between cards

5. **Real-Time Updates**
   - WebSocket for live status
   - Frontend updates automatically
   - No polling needed

---

## Development Workflow

### Frontend
```bash
cd frontend
npm install
npm run dev  # Runs on :3000
```

### Backend
```bash
cd backend
source .venv/bin/activate  # or use uv
pip install -e .
uvicorn app.main:app --reload  # Runs on :8000
```

### Modal Deployment
```bash
cd backend
source .venv/bin/activate
modal deploy cards/modal_app.py
```

---

## Adding a New Card

1. **Create card class** in `backend/cards/`
   - Extend `BaseCard`
   - Implement `execute()` and `get_output_preview()`
   - Define schemas (config, input, output)

2. **Register card** in `backend/cards/registry.py`
   - Add `_register(YourCard())`

3. **Card appears automatically** in frontend palette
   - No frontend changes needed!

---

## Future Enhancements

- GPU support for deep learning cards
- S3 storage for production
- Authentication and multi-user support
- Pipeline versioning and history
- Export/import pipelines
- More card types (preprocessing, feature engineering, etc.)
