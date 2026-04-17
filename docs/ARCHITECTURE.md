# SkyPark Central — Architecture

This document describes the end-to-end architecture of the SkyPark
Central parking chatbot after Stage 4. For a quickstart, see the
top-level `README.md`; this document is the reference for agents and
engineers working on the system.

---

## 1. High-level diagram

```
                             ┌────────────────────────────┐
                             │   Stage 4 Orchestrator     │
                             │   LangGraph StateGraph     │
                             │                            │
                             │   user_node ──▶ approval   │
                             │      │          (interrupt)│
                             │      ▼          ▼          │
                             │     END ◀── decision_node  │
                             │                 │          │
                             │                 ▼          │
                             │         recording_node ──▶ │
                             │                            │
                             └───────┬────────────────────┘
                                     │ (same process)
     ┌───────────────────┐           │
     │   User            │           │
     │   chatbot CLI     │───────────┤
     │   (Stage 1 agent) │           │
     └───────────────────┘           │
                                     │
     ┌───────────────────┐           │
     │  Admin dashboard  │           │
     │  + admin CLI      │───────────┼──▶ ┌─────────────────────┐
     │  (Stage 2 agent)  │           │    │      SQLite         │
     └───────────────────┘           │    │  hours / pricing /  │
                                     │    │  spots / bookings   │
                                     │    └─────────────────────┘
                                     │              │
                                     ▼              │
                           ┌──────────────────┐     │
                           │  MCP Server      │◀────┘
                           │  (Stage 3)       │  on approval
                           │  JSON-RPC 2.0    │
                           │  Bearer auth     │
                           └─────────┬────────┘
                                     │
                                     ▼
                        data/confirmed_reservations.txt
                        Name | Car | Period | Approved At
```

Three separate user-facing entry points all converge on the same
SQLite `bookings` table, then fan out to the MCP server on approval:

1. **User chatbot** (CLI or REST) — stages a booking as `pending`.
2. **Admin** (dashboard / CLI / REST / natural-language agent) — marks
   it `confirmed` or `rejected`.
3. **Orchestrator** (single-process LangGraph demo) — runs steps 1 and
   2 in the same graph with a `MemorySaver` checkpoint and an
   `interrupt` boundary between them.

On `approved → confirmed`, every write path calls `mcp_client.write_confirmed_reservation(booking)`, which invokes the MCP server over HTTP.

---

## 2. Components

### 2.1 Stage 1 — User chatbot (`src/chatbot.py`)

A LangGraph ReAct agent (`create_react_agent`) with six tools:

| Tool | Backing | Purpose |
|---|---|---|
| `search_parking_info` | Pinecone RAG | Static facility info (policies, location, amenities) |
| `get_working_hours`   | SQLite | Live hours |
| `get_pricing`         | SQLite | Live pricing table |
| `check_availability`  | SQLite | Free spots per zone |
| `create_reservation`  | SQLite | Stage a PENDING booking + notify admin |
| `check_reservation_status` | SQLite | Read booking status by ID |

Guardrails (`src/guardrails.py`) wrap the agent:
- `sanitize_input` runs on the user message before the LLM sees it
  (PII redaction, prompt-injection detection).
- `sanitize_output` runs on the final reply (secret / PII leakage).

### 2.2 Stage 2 — Admin path

Three entry points, one agent:

- **Dashboard** — `GET /admin` renders pending + all bookings with
  Approve/Reject buttons (`src/server.py`).
- **Admin CLI** — `python3 -m src.admin_cli` — `/pending`,
  `/approve <id>`, `/reject <id>`, plus free-text to the admin agent.
- **Admin agent** — LangGraph ReAct agent with six tools
  (`list_pending_reservations`, `inspect_reservation`,
  `check_availability`, `approve_reservation`, `reject_reservation`,
  `view_notifications`). Used from the dashboard chat widget or
  `POST /api/admin/chat`.
- **REST API** — `POST /api/admin/bookings/{id}/approve` and
  `/reject`.

All approval paths funnel through `db.approve_booking(id, notes)` →
notifications → `mcp_client.write_confirmed_reservation(booking)`.

Notifications (`src/notifications.py`) are in-app (memory-backed) and
optionally SMTP email if `SMTP_*` env vars are set.

### 2.3 Stage 3 — MCP server

Implemented in pure FastAPI + JSON-RPC 2.0 at `POST /mcp`
(`src/mcp_server.py`). No dependency on the `mcp` SDK, so Python 3.9
is supported.

- **Auth**: every request must carry `Authorization: Bearer <MCP_SECRET>`.
  Missing or wrong token → HTTP 401.
- **Methods**: `initialize`, `tools/list`, `tools/call`.
- **Tool**: `write_confirmed_reservation` delegates to the pure
  `src/reservation_writer.py` module, which uses `fcntl.flock(LOCK_EX)`
  for concurrent-safe appends and sanitises pipe characters (`|`→`/`)
  to prevent format injection.

File format (one line per booking):
```
Name | Car Number | 2026-04-20T09:00 - 2026-04-20T18:00 | 2026-04-20T08:30:00Z
```

Client side (`src/mcp_client.py`) is a synchronous `httpx` wrapper.
If the server is unreachable, the client logs a warning and returns a
string — it does not raise — so approval never fails because the MCP
server is down.

### 2.4 Stage 4 — Orchestrator (`src/orchestrator.py`)

A single LangGraph `StateGraph` that binds the three stages into one
pipeline with an explicit HITL boundary.

**State**

```python
class OrchestratorState(TypedDict, total=False):
    user_message: str
    assistant_reply: str
    thread_id: str
    booking_id: Optional[int]
    booking: Optional[dict]
    decision: Optional[dict]       # {"action": "approve"|"reject", "notes": str}
    mcp_result: Optional[str]
    final_status: str              # "confirmed" | "rejected" | "conversation" | ...
```

**Nodes**

| Node | Responsibility | Side effects |
|---|---|---|
| `user_node` | Call Stage 1 chatbot; detect whether a booking was staged via `list_bookings` diff | SQL insert (via chatbot tool) |
| `approval_node` | Emit payload via `interrupt(...)` — graph pauses; on resume, return admin decision | None before `interrupt` (safe for replay) |
| `decision_node` | Call `db.approve_booking` or `db.reject_booking` | SQL update |
| `recording_node` | Call `mcp_write(booking)` with graceful error handling | HTTP POST to MCP server |

**Edges**

```
START → user_node
user_node → (booking_id?) → approval_node | END
approval_node → decision_node
decision_node → (confirmed?) → recording_node | END
recording_node → END
```

**Checkpointing**

`MemorySaver()` — in-memory, keyed by `thread_id`. Each conversation
gets its own thread, so the graph can pause at `approval_node` for one
thread while other threads run independently. For production, swap in
`SqliteSaver` or `PostgresSaver`.

**Public API**

- `build_orchestrator()` → compiled graph
- `start_turn(graph, user_message, thread_id)` — invokes up to the
  first interrupt; returns `{interrupted, payload, assistant_reply}`
- `resume_with_decision(graph, thread_id, action, notes)` — sends
  `Command(resume={...})`; returns `{final_status, booking, mcp_result}`

**Why the orchestrator complements rather than replaces the async path**

The async path (user session in one CLI, admin session in another, minutes or hours apart) is the realistic production flow. The orchestrator is valuable because it:

1. Demonstrates the full pipeline using LangGraph idioms
   (`StateGraph`, `interrupt`, `Command`) — useful for the assignment
   rubric and for reviewers reading the codebase.
2. Makes integration testing tractable — one process, deterministic,
   stubbable.
3. Embeds cleanly in other orchestrators (notebooks, FastAPI routes,
   background jobs).

---

## 3. Data model

SQLite schema (`src/db.py`):

- `hours(day, open_time, close_time, is_24h)`
- `pricing(vehicle_type, duration, zone, price_usd)`
- `spots(zone, total, occupied)` — `available = total - occupied`
- `bookings(id, first_name, last_name, car_plate, start_ts, end_ts, status, admin_notes, reviewed_at, created_at)`

Booking state machine:

```
 (created)
     │
     ▼
  pending ──▶ confirmed  ──(mcp_write)──▶  data/confirmed_reservations.txt
     │
     └────▶ rejected
```

Transitions happen only via `db.approve_booking(id, notes)` and
`db.reject_booking(id, notes)`. Both require `status='pending'` in
the `WHERE` clause, so a booking can be approved or rejected at most
once.

---

## 4. Security

- **MCP server**: Bearer token. Rotate `MCP_SECRET` in `.env` before
  any real deployment; the shipped default is explicitly a
  placeholder.
- **Guardrails**: user input is redacted before the LLM sees it, and
  the reply is re-checked before it goes back to the user.
- **Format injection**: `reservation_writer.py` sanitises `|` to `/`
  in every field before appending.
- **Concurrency**: `fcntl.flock` exclusive lock serialises writes to
  the output file; SQLite serialises DB writes with its own lock.
- **Admin surface**: the admin dashboard + agent are internal. In a
  real deployment they must sit behind auth (HTTP basic, SSO, etc.).
  This project ships them unauthenticated for local development.

---

## 5. Setup & deployment

### 5.1 Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY, PINECONE_API_KEY, and rotate MCP_SECRET

python3 -m src.db                    # initialise SQLite
python3 -m src.ingest                # populate Pinecone
```

### 5.2 Running the full stack

Open three terminals:

| Terminal | Command | Port |
|---|---|---|
| 1 | `python3 -m src.mcp_server` | 8001 |
| 2 | `python3 -m src.server`     | 8000 |
| 3 | `python3 -m src.cli` or `python3 -m src.orchestrator_cli` | — |

Users hit terminal 3 (chatbot); admins open `http://localhost:8000/admin`
or terminal `python3 -m src.admin_cli`.

### 5.3 Production notes

- Put both HTTP services (8000, 8001) behind a reverse proxy (nginx /
  Caddy) with TLS.
- Replace `MemorySaver` in both agents and the orchestrator with
  `SqliteSaver` or `PostgresSaver` so interrupted threads survive
  restarts.
- Replace SQLite with Postgres for multi-worker deployments.
- Rotate `MCP_SECRET`; consider mTLS between the approval service and
  the MCP server.
- Configure a real SMTP sender or switch `notifications.py` to a
  transactional mail API.

---

## 6. Testing

### 6.1 Unit + integration

```bash
python3 -m pytest tests/ -v
```

- `tests/test_guardrails.py` — input/output sanitisation.
- `tests/test_mcp_server.py` — the pure `reservation_writer` module.
- `tests/test_orchestrator.py` — end-to-end through the compiled
  LangGraph, with `user_chat` and `mcp_write` stubbed.

### 6.2 Load tests

```bash
python3 -m scripts.load_test db     # SQLite approval contention
python3 -m scripts.load_test mcp    # MCP server under parallel writes (requires server running)
python3 -m scripts.load_test orch   # LangGraph + checkpointer throughput
python3 -m scripts.load_test all    # all three
```

Each scenario reports p50 / p95 / p99 / max latency and throughput.
Tune `--total` and `--workers` to match your environment.

### 6.3 Retrieval evaluation

```bash
python3 -m eval.evaluate
python3 -m eval.evaluate --end-to-end
```

Scores Recall@K, Precision@K, MRR, and end-to-end chatbot latency
against `eval/questions.json`.

---

## 7. Directory reference

```
src/
  config.py                # env-driven Settings
  db.py                    # SQLite schema + CRUD
  guardrails.py            # input/output sanitisation
  retriever.py             # Pinecone vector store factory
  ingest.py                # markdown → Pinecone

  chatbot.py               # Stage 1 — user LangGraph agent
  cli.py                   # Stage 1 — user REPL

  notifications.py         # Stage 2 — in-app + SMTP
  admin_agent.py           # Stage 2 — admin LangGraph agent
  admin_cli.py             # Stage 2 — admin REPL
  server.py                # Stage 2 — FastAPI dashboard + REST API

  reservation_writer.py    # Stage 3 — pure file writer (fcntl lock)
  mcp_server.py            # Stage 3 — FastAPI JSON-RPC server
  mcp_client.py            # Stage 3 — httpx client

  orchestrator.py          # Stage 4 — LangGraph StateGraph
  orchestrator_cli.py      # Stage 4 — single-process demo

scripts/
  load_test.py             # Stage 4 — load harness

docs/
  ARCHITECTURE.md          # this file

tests/
  test_guardrails.py
  test_mcp_server.py
  test_orchestrator.py

eval/
  questions.json
  evaluate.py

data/
  static/                  # markdown KB for Pinecone
  dynamic/parking.db       # SQLite (created on first run)
  confirmed_reservations.txt  # MCP output (gitignored)
```
