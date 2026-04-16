# Parking Chatbot — Stage 1

A Retrieval-Augmented Generation (RAG) chatbot for the fictional **SkyPark
Central** parking facility. Built with **LangChain**, **LangGraph**,
**Pinecone**, and **OpenAI GPT**.

This repository implements **Stage 1** of the project: a working chatbot that
answers questions about the facility, queries live operational data, stages
reservations for human-in-the-loop approval, applies safety guardrails to
input and output, and ships with an evaluation harness.

---

## Architecture

```
                      ┌──────────────────────────┐
        user input -> │  guardrails.sanitize_in  │ -> blocked? -> refusal
                      └──────────────┬───────────┘
                                     │ (cleaned text)
                                     ▼
                      ┌──────────────────────────┐
                      │   LangGraph ReAct agent  │ <— OpenAI GPT
                      │      (chatbot.py)        │
                      └──────────────┬───────────┘
                                     │ tool calls
       ┌─────────────────────────────┼────────────────────────────────┐
       ▼                             ▼                                ▼
┌──────────────┐         ┌──────────────────────┐         ┌────────────────────┐
│  Pinecone    │         │  SQLite (db.py)       │         │ create_reservation │
│ vector store │         │  hours / pricing /    │         │  -> bookings table │
│ (static MD)  │         │  spots availability   │         │   status=pending   │
└──────────────┘         └──────────────────────┘         └────────────────────┘
       ▲                                                          (HITL → Stage 2)
       │ ingest.py
┌──────────────┐
│ data/static/ │  ← markdown KB (general, location, booking, policies, hours, FAQ)
└──────────────┘

                                     │
                                     ▼
                      ┌──────────────────────────┐
                      │ guardrails.sanitize_out  │
                      └──────────────┬───────────┘
                                     ▼
                                  reply
```

**Static vs dynamic split** (the brief's optional improvement):
- **Static** facility info (location, amenities, policies, booking process,
  FAQ) lives in `data/static/*.md` and is embedded into Pinecone.
- **Dynamic** operational data (working hours, prices, live availability,
  bookings) lives in SQLite at `data/dynamic/parking.db`. Tools query it
  directly so answers are always fresh.

---

## Project layout

```
├── data/
│   ├── static/                  # markdown KB → vector store
│   │   ├── general.md
│   │   ├── location.md
│   │   ├── booking_process.md
│   │   ├── policies.md
│   │   ├── hours_and_pricing_overview.md
│   │   └── faq.md
│   └── dynamic/                 # SQLite db (created on first run)
├── src/
│   ├── config.py                # env-var settings
│   ├── db.py                    # SQLite schema + seed + read helpers
│   ├── guardrails.py            # PII / secret / prompt-injection filter
│   ├── ingest.py                # markdown → Pinecone upsert
│   ├── retriever.py             # cached vector-store retriever
│   ├── chatbot.py               # LangGraph ReAct agent + tools
│   └── cli.py                   # interactive REPL
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

### 1. Create a virtual environment and install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# then edit .env and fill in:
#   OPENAI_API_KEY=...
#   PINECONE_API_KEY=...
```

The defaults in `.env.example` use:
- **OpenAI** model `gpt-4o-mini` (override via `OPENAI_MODEL`)
- **Pinecone** serverless on `aws / us-east-1` (override via `PINECONE_*`)
- **HuggingFace** local embeddings `all-MiniLM-L6-v2` (384-dim, no API key)

### 3. Initialise the SQLite dynamic store

```bash
python -m src.db
```

### 4. Ingest the static knowledge base into Pinecone

```bash
python -m src.ingest                  # incremental upsert
python -m src.ingest --recreate       # drop & recreate the index
```

### 5. Chat

```bash
python -m src.cli
```

```
SkyPark Central — Parking Chatbot (Stage 1)
Type your question, or '/quit' to exit, '/new' for a new conversation.

you > Where is the garage and is it open at night?
bot > SkyPark Central is at 120 Harbor Avenue, Rivertown ... and yes, it's open 24/7.
you > How many spaces are free on L3 right now?
bot > L3 currently has 59 of 100 spaces free.
you > I'd like to book a spot
bot > Sure — could I have your first and last name, license plate, and the start/end of your reservation?
```

---

## Guardrails

`src/guardrails.py` runs at the chatbot boundary on **every** turn:

- **Input**:
  - Blocks empty messages.
  - Blocks obvious prompt-injection patterns
    (`ignore previous instructions`, `reveal system prompt`, …).
  - Redacts emails, phone numbers, SSNs, credit-card numbers, IBANs, JWTs,
    and common API-key shapes (`sk-…`, `AKIA…`, `ghp_…`).
  - Flags secret-related keywords (`password`, `api key`, …).
- **Output**:
  - Same redaction pass on the model reply, so any sensitive text that may
    have crept into the vector DB (or that the model invented) cannot reach
    the user.

The choice of regex-first detection is deliberate for Stage 1: it has no
extra dependencies, is fully transparent, and errs on over-redaction. For
production the same interface can be backed by Microsoft Presidio or a
similar NLP-based PII detector — the call sites in `chatbot.py` will not
need to change.

Run the smoke tests:

```bash
pip install pytest
pytest tests/
```

---

## Evaluation

`eval/questions.json` is a hand-labelled set of 18 user questions, each
tagged with the **gold topics** (markdown topics that should be retrieved).
`eval/evaluate.py` computes:

| Metric | Definition |
|---|---|
| **Recall@K**    | Fraction of questions for which ≥ 1 retrieved chunk is from a gold topic. |
| **Precision@K** | Average per-question fraction of retrieved chunks whose topic is gold. |
| **MRR@K**       | Mean reciprocal rank of the first gold-topic chunk. |
| **Retrieval latency** | p50 / p95 / mean wall-clock per `retriever.invoke()` call. |
| **End-to-end latency** | p50 / p95 / mean wall-clock per full `chat()` call (with `--end-to-end`). |

Run it:

```bash
python -m eval.evaluate                   # retrieval-only (cheap, no LLM calls)
python -m eval.evaluate --end-to-end      # also time full chatbot turns
python -m eval.evaluate --k 6             # change top-k
```

The harness writes `eval/results/report.md` (human-readable) and
`eval/results/report.json` (machine-readable). The retrieval-only run uses
no OpenAI credits; only `--end-to-end` calls the LLM.

---

## Reservation flow (Stage 1 scope)

The agent collects four fields — first name, last name, license plate, and
start/end timestamps — and inserts a booking row with `status='pending'`.
The actual **human-in-the-loop confirmation** (an admin reviewing and
flipping the status to `confirmed`) is the subject of **Stage 2** and is
intentionally not implemented here. No payment information is ever
requested or stored.

You can inspect staged bookings any time:

```bash
python -c "from src.db import list_bookings; import json; print(json.dumps(list_bookings(), indent=2))"
```

---

## What's intentionally NOT in Stage 1

- Human-in-the-loop reservation approval workflow (Stage 2)
- A web/HTTP frontend (Stage 3)
- Payment processing (out of scope for the entire project)

---

## License

For educational use as part of the parking-chatbot course project.
