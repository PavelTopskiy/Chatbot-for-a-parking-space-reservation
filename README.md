# Parking Chatbot — Stage 1 + Stage 2 + Stage 3

A Retrieval-Augmented Generation (RAG) chatbot for the fictional **SkyPark
Central** parking facility. Built with **LangChain**, **LangGraph**,
**Pinecone**, and **OpenAI GPT**.

- **Stage 1** — RAG chatbot, vector + SQL data split, guardrails, evaluation
- **Stage 2** — Human-in-the-loop reservation approval with admin agent,
  REST API, dashboard, and email notifications
- **Stage 3** — MCP server that writes confirmed reservations to a file

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │                    USER SIDE                             │
  │                                                         │
  │  user ──> guardrails ──> User Agent (chatbot.py)        │
  │                          6 tools:                       │
  │                          - search_parking_info (RAG)    │
  │                          - get_working_hours (SQL)      │
  │                          - get_pricing (SQL)            │
  │                          - check_availability (SQL)     │
  │                          - create_reservation (SQL)  ───┼──> notifications.py
  │                          - check_reservation_status     │        │
  │           guardrails <── reply                          │        │
  └─────────────────────────────────────────────────────────┘        │
                                                                     │
                                                            ┌────────▼────────┐
                                                            │  Notification   │
                                                            │  in-app + email │
                                                            └────────┬────────┘
                                                                     │
  ┌──────────────────────────────────────────────────────────────────┐│
  │                    ADMIN SIDE                                    ││
  │                                                                 ││
  │  ┌─────────────────────────┐    ┌───────────────────────────┐   ││
  │  │  Admin Dashboard (HTML) │    │  Admin CLI (admin_cli.py) │   ││
  │  │  GET /admin             │    │  /pending /approve /reject│   ││
  │  └──────────┬──────────────┘    └──────────┬────────────────┘   ││
  │             │                              │                    ││
  │             ▼                              ▼                    ││
  │  ┌──────────────────────────────────────────────────────┐       ││
  │  │          FastAPI Server (server.py)                   │<─────┘│
  │  │  POST /api/admin/bookings/{id}/approve               │       │
  │  │  POST /api/admin/bookings/{id}/reject                │       │
  │  │  POST /api/admin/chat  ──> Admin Agent                │       │
  │  └──────────────────────────────────────────────────────┘       │
  │                                    │                            │
  │               Admin Agent (admin_agent.py)                      │
  │               6 tools: list_pending, inspect, availability,     │
  │                        approve, reject, view_notifications      │
  └─────────────────────────────────────────────────────────────────┘
                                    │
                           ┌────────▼────────┐
                           │     SQLite      │
                           │  bookings table │
                           │  status: pending│
                           │  → confirmed    │
                           │  → rejected     │
                           └─────────────────┘
```

---

## Project layout

```
├── data/
│   ├── static/                  # markdown KB → vector store
│   └── dynamic/                 # SQLite db (created on first run)
├── src/
│   ├── config.py                # env-var settings
│   ├── db.py                    # SQLite schema + booking management
│   ├── guardrails.py            # PII / secret / prompt-injection filter
│   ├── ingest.py                # markdown → Pinecone upsert
│   ├── retriever.py             # cached vector-store retriever
│   ├── chatbot.py               # User-facing LangGraph agent (6 tools)
│   ├── cli.py                   # User interactive REPL
│   ├── admin_agent.py           # Admin-facing LangGraph agent (6 tools)
│   ├── admin_cli.py             # Admin interactive REPL
│   ├── notifications.py         # In-app + email notification service
│   ├── server.py                # FastAPI: REST API + admin dashboard
│   ├── reservation_writer.py    # Pure file-write logic (no external deps)
│   ├── mcp_server.py            # MCP server (JSON-RPC, port 8001, bearer auth)
│   └── mcp_client.py            # Sync MCP client used by approval paths
├── eval/
│   ├── questions.json           # gold QA set with topic labels
│   └── evaluate.py              # Recall@K, Precision@K, MRR, latency
├── tests/
│   └── test_guardrails.py
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Credentials

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and PINECONE_API_KEY
```

### 3. Initialise database and vector store

```bash
python3 -m src.db                     # seed SQLite
python3 -m src.ingest                 # populate Pinecone
```

---

## Running

### User chatbot (CLI)

```bash
python3 -m src.cli
```

```
you > I'd like to reserve a spot
bot > Sure! I'll need your first name, last name, license plate, and ...
you > John Doe, ABC-1234, April 20 9am to 6pm
bot > Reservation #1 staged as PENDING. The administrator has been notified.
you > What's the status of reservation 1?
bot > Reservation #1: PENDING — awaiting admin review.
```

### Admin dashboard (web)

```bash
python3 -m src.server
```

Open **http://localhost:8000/admin** in your browser. The dashboard shows:
- **Pending reservations** with Approve / Reject buttons
- **All reservations** with status badges
- **Admin AI assistant** chat for natural-language review

### Admin CLI

```bash
python3 -m src.admin_cli
```

```
admin > /pending
  #1 | John Doe | ABC-1234 | 2026-04-20T09:00 -> 2026-04-20T18:00
admin > /approve 1
  Notes (optional): looks good
  Reservation #1 CONFIRMED.
admin > Should I approve reservation #2? Check availability first.
assistant > Let me check... L1 has 38 free spaces. The reservation looks
            feasible. I recommend approving it.
```

### REST API

```bash
# List pending bookings
curl http://localhost:8000/api/admin/bookings?status=pending

# Approve a booking
curl -X POST http://localhost:8000/api/admin/bookings/1/approve \
  -H 'Content-Type: application/json' \
  -d '{"notes": "approved"}'

# Reject a booking
curl -X POST http://localhost:8000/api/admin/bookings/1/reject \
  -H 'Content-Type: application/json' \
  -d '{"reason": "garage full on that date"}'

# User checks status
curl http://localhost:8000/api/booking/1

# Chat with admin agent
curl -X POST http://localhost:8000/api/admin/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "show me all pending reservations and check availability"}'
```

---

## Stage 2: Human-in-the-Loop flow

### How it works

1. **User creates a reservation** via the chatbot (CLI or `/api/chat`).
2. The chatbot inserts a row with `status='pending'` into SQLite.
3. `notifications.py` fires:
   - Always: an **in-app notification** (visible in dashboard + admin CLI).
   - Optionally: an **SMTP email** to the admin (if `SMTP_*` env vars are set).
4. The **admin** reviews the reservation via:
   - The **web dashboard** (`/admin`) — click Approve or Reject.
   - The **admin CLI** (`python3 -m src.admin_cli`) — `/approve ID` or `/reject ID`.
   - The **REST API** — `POST /api/admin/bookings/{id}/approve` or `/reject`.
   - The **admin AI agent** — natural-language instruction like
     "approve reservation 1" (the agent checks availability and recommends).
5. On approval/rejection, a notification is generated (in-app + email).
6. The **user** can check status via the chatbot ("what's the status of
   reservation 1?") or `GET /api/booking/{id}`.

### Email notifications (optional)

To enable email, add SMTP settings to `.env`:

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
ADMIN_EMAIL=admin@skypark-central.example
```

For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833).
If SMTP is not configured, the system works fine — notifications are in-app only.

---

## Guardrails

See Stage 1 docs above. Guardrails apply to the user chatbot only (the admin
agent is an internal tool, not user-facing).

```bash
python3 -m pytest tests/
```

---

## Evaluation

```bash
python3 -m eval.evaluate                   # retrieval-only
python3 -m eval.evaluate --end-to-end      # full chatbot turns
```

---

## Stage 3: MCP Server

### Architecture

```
  Admin approves reservation
         │
         ▼
  mcp_client.write_confirmed_reservation(booking)
         │  (POST /mcp JSON-RPC + Bearer token)
         ▼
  MCP Server  (src/mcp_server.py, port 8001)
  ┌──────────────────────────────────────────┐
  │  BearerAuthMiddleware                    │
  │  → validates Authorization: Bearer TOKEN │
  │                                          │
  │  Tool: write_confirmed_reservation       │
  │  → delegates to reservation_writer.py   │
  │  → fcntl file lock                      │
  │  → appends to confirmed_reservations.txt │
  └──────────────────────────────────────────┘
         │
         ▼
  data/confirmed_reservations.txt
  Name | Car Number | Reservation Period | Approval Time
```

### File format

```
Ivan Petrenko | AA1234BB | 2026-04-20T09:00 - 2026-04-20T18:00 | 2026-04-20T08:30:00Z
Anna Koval    | BB5678CC | 2026-04-21T10:00 - 2026-04-21T17:00 | 2026-04-21T09:15:00Z
```

### Running the MCP server

Start it **before** the admin dashboard (it runs on port 8001):

```bash
# Terminal 3
source .venv/bin/activate
python3 -m src.mcp_server
```

```
INFO: Starting SkyPark MCP server on 0.0.0.0:8001
INFO: Reservations file: .../data/confirmed_reservations.txt
INFO: Auth: enabled
```

Now when an admin approves a reservation (via dashboard, admin CLI, or admin
agent), the MCP client is called automatically and the entry is written.

### Security

- Every request to the MCP server must include `Authorization: Bearer <MCP_SECRET>`.
- The secret is set via the `MCP_SECRET` env var (default in `.env.example` is
  a placeholder — **change it before deploying**).
- Requests without a valid token receive HTTP 401.
- All input fields are sanitised (pipe characters replaced with `/`) before
  writing to prevent format injection.
- File writes use `fcntl.flock` exclusive locking to prevent race conditions.

### Complete 3-server workflow

| Terminal | Command | Purpose |
|---|---|---|
| 1 | `python3 -m src.mcp_server` | MCP server (port 8001) |
| 2 | `python3 -m src.server`     | REST API + dashboard (port 8000) |
| 3 | `python3 -m src.cli`        | User chatbot REPL |

Or use the admin CLI instead of the dashboard:
```bash
python3 -m src.admin_cli   # /approve, /reject, /pending
```

After approval, check the output file:
```bash
cat data/confirmed_reservations.txt
```

---

## License

For educational use as part of the parking-chatbot course project.
