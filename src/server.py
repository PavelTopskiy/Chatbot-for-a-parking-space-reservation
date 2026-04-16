"""FastAPI server exposing the chatbot and admin dashboard.

Endpoints:

  User-facing:
    POST /api/chat              — send a user message, get chatbot reply
    GET  /api/booking/{id}      — check booking status

  Admin REST API:
    GET  /api/admin/bookings              — list bookings (filter by status)
    GET  /api/admin/bookings/{id}         — inspect one booking
    POST /api/admin/bookings/{id}/approve — approve a pending booking
    POST /api/admin/bookings/{id}/reject  — reject a pending booking
    GET  /api/admin/notifications         — list notifications
    POST /api/admin/chat                  — talk to the admin agent

  Admin dashboard:
    GET  /admin                           — HTML dashboard

Run::

    python3 -m src.server
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import db
from . import notifications as notif
from .admin_agent import admin_chat
from .chatbot import chat

app = FastAPI(title="SkyPark Central Parking Chatbot", version="2.0")


# ---------- request/response models ----------

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    input_findings: list
    output_findings: list
    blocked: bool

class AdminChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ApproveRequest(BaseModel):
    notes: Optional[str] = ""

class RejectRequest(BaseModel):
    reason: str


# ---------- user endpoints ----------

@app.post("/api/chat", response_model=ChatResponse)
def user_chat(req: ChatRequest):
    tid = req.thread_id or uuid.uuid4().hex[:8]
    result = chat(req.message, thread_id=tid)
    return ChatResponse(thread_id=tid, **result)


@app.get("/api/booking/{booking_id}")
def get_booking_status(booking_id: int):
    b = db.get_booking(booking_id)
    if not b:
        raise HTTPException(404, f"Booking #{booking_id} not found")
    return {"id": b["id"], "status": b["status"], "admin_notes": b.get("admin_notes")}


# ---------- admin REST API ----------

@app.get("/api/admin/bookings")
def list_bookings(status: Optional[str] = Query(None)):
    return db.list_bookings(status=status)


@app.get("/api/admin/bookings/{booking_id}")
def inspect_booking(booking_id: int):
    b = db.get_booking(booking_id)
    if not b:
        raise HTTPException(404, f"Booking #{booking_id} not found")
    return b


@app.post("/api/admin/bookings/{booking_id}/approve")
def approve_booking(booking_id: int, req: ApproveRequest):
    ok = db.approve_booking(booking_id, admin_notes=req.notes or "")
    if not ok:
        raise HTTPException(400, "Booking not found or not pending")
    booking = db.get_booking(booking_id)
    notif.notify_booking_confirmed(booking)
    return {"status": "confirmed", "booking": booking}


@app.post("/api/admin/bookings/{booking_id}/reject")
def reject_booking(booking_id: int, req: RejectRequest):
    if not req.reason.strip():
        raise HTTPException(400, "Reason is required")
    ok = db.reject_booking(booking_id, admin_notes=req.reason)
    if not ok:
        raise HTTPException(400, "Booking not found or not pending")
    booking = db.get_booking(booking_id)
    notif.notify_booking_rejected(booking)
    return {"status": "rejected", "booking": booking}


@app.get("/api/admin/notifications")
def list_notifications(unread_only: bool = False):
    return notif.get_notifications(unread_only=unread_only)


@app.post("/api/admin/chat")
def admin_chat_endpoint(req: AdminChatRequest):
    tid = req.thread_id or "admin-" + uuid.uuid4().hex[:8]
    reply = admin_chat(req.message, thread_id=tid)
    return {"reply": reply, "thread_id": tid}


# ---------- admin dashboard ----------

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SkyPark Central — Admin Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; color: #333; }
  .header { background: #1a1a2e; color: #fff; padding: 1rem 2rem; }
  .header h1 { font-size: 1.3rem; }
  .container { max-width: 960px; margin: 1rem auto; padding: 0 1rem; }
  .card { background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          margin-bottom: 1rem; padding: 1.2rem; }
  .card h2 { font-size: 1.1rem; margin-bottom: .8rem; color: #1a1a2e; }
  table { width: 100%; border-collapse: collapse; font-size: .9rem; }
  th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid #eee; }
  th { background: #f9f9f9; font-weight: 600; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: .8rem; font-weight: 600; }
  .badge-pending { background: #fff3cd; color: #856404; }
  .badge-confirmed { background: #d4edda; color: #155724; }
  .badge-rejected { background: #f8d7da; color: #721c24; }
  .btn { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer;
         font-size: .85rem; font-weight: 500; }
  .btn-approve { background: #28a745; color: #fff; }
  .btn-reject { background: #dc3545; color: #fff; }
  .btn:hover { opacity: .85; }
  #admin-chat { display: flex; gap: .5rem; margin-top: .8rem; }
  #admin-chat input { flex: 1; padding: .5rem; border: 1px solid #ccc; border-radius: 4px; }
  #admin-chat button { padding: .5rem 1rem; }
  #chat-log { max-height: 300px; overflow-y: auto; font-size: .85rem;
              background: #fafafa; padding: .8rem; border-radius: 4px;
              margin-top: .5rem; white-space: pre-wrap; }
  .refresh-btn { float: right; font-size: .85rem; cursor: pointer;
                 color: #1a1a2e; text-decoration: underline; border: none;
                 background: none; }
</style>
</head>
<body>
<div class="header"><h1>SkyPark Central — Admin Dashboard</h1></div>
<div class="container">

  <div class="card">
    <h2>Pending Reservations <button class="refresh-btn" onclick="loadBookings()">refresh</button></h2>
    <table>
      <thead><tr><th>ID</th><th>Guest</th><th>Plate</th><th>Period</th><th>Created</th><th>Actions</th></tr></thead>
      <tbody id="bookings-body"><tr><td colspan="6">Loading...</td></tr></tbody>
    </table>
  </div>

  <div class="card">
    <h2>All Reservations <button class="refresh-btn" onclick="loadAllBookings()">refresh</button></h2>
    <table>
      <thead><tr><th>ID</th><th>Guest</th><th>Plate</th><th>Period</th><th>Status</th><th>Notes</th></tr></thead>
      <tbody id="all-bookings-body"><tr><td colspan="6">Loading...</td></tr></tbody>
    </table>
  </div>

  <div class="card">
    <h2>Admin AI Assistant</h2>
    <div id="chat-log">Ask the assistant to review, summarize, or help decide on reservations.</div>
    <div id="admin-chat">
      <input id="chat-input" placeholder="e.g. Show me pending reservations and check availability" onkeydown="if(event.key==='Enter')sendChat()">
      <button class="btn btn-approve" onclick="sendChat()">Send</button>
    </div>
  </div>
</div>

<script>
const API = '';
let chatThreadId = null;

async function loadBookings() {
  const res = await fetch(API + '/api/admin/bookings?status=pending');
  const data = await res.json();
  const tbody = document.getElementById('bookings-body');
  if (!data.length) { tbody.innerHTML = '<tr><td colspan="6">No pending reservations</td></tr>'; return; }
  tbody.innerHTML = data.map(b => `<tr>
    <td>${b.id}</td>
    <td>${b.first_name} ${b.last_name}</td>
    <td>${b.car_plate}</td>
    <td>${b.start_ts} &rarr; ${b.end_ts}</td>
    <td>${b.created_at}</td>
    <td>
      <button class="btn btn-approve" onclick="approveBooking(${b.id})">Approve</button>
      <button class="btn btn-reject" onclick="rejectBooking(${b.id})">Reject</button>
    </td>
  </tr>`).join('');
}

async function loadAllBookings() {
  const res = await fetch(API + '/api/admin/bookings');
  const data = await res.json();
  const tbody = document.getElementById('all-bookings-body');
  if (!data.length) { tbody.innerHTML = '<tr><td colspan="6">No reservations</td></tr>'; return; }
  tbody.innerHTML = data.map(b => {
    const cls = b.status === 'pending' ? 'badge-pending' : b.status === 'confirmed' ? 'badge-confirmed' : 'badge-rejected';
    return `<tr>
      <td>${b.id}</td>
      <td>${b.first_name} ${b.last_name}</td>
      <td>${b.car_plate}</td>
      <td>${b.start_ts} &rarr; ${b.end_ts}</td>
      <td><span class="badge ${cls}">${b.status}</span></td>
      <td>${b.admin_notes || ''}</td>
    </tr>`;
  }).join('');
}

async function approveBooking(id) {
  const notes = prompt('Optional notes for approval:', '');
  if (notes === null) return;
  await fetch(API + `/api/admin/bookings/${id}/approve`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({notes})
  });
  loadBookings(); loadAllBookings();
}

async function rejectBooking(id) {
  const reason = prompt('Reason for rejection (required):');
  if (!reason) return;
  await fetch(API + `/api/admin/bookings/${id}/reject`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({reason})
  });
  loadBookings(); loadAllBookings();
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  const log = document.getElementById('chat-log');
  log.textContent += '\\nAdmin > ' + msg;
  input.value = '';
  const res = await fetch(API + '/api/admin/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: msg, thread_id: chatThreadId})
  });
  const data = await res.json();
  chatThreadId = data.thread_id;
  log.textContent += '\\nAssistant > ' + data.reply + '\\n';
  log.scrollTop = log.scrollHeight;
}

loadBookings();
loadAllBookings();
</script>
</body>
</html>
"""


@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard():
    return DASHBOARD_HTML


# ---------- startup ----------

@app.on_event("startup")
def on_startup():
    db.init_db()


def main():
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
