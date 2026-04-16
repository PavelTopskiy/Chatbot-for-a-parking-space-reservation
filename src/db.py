"""SQLite layer for *dynamic* parking data: hours, pricing, availability, bookings.

Static facility information lives in the vector database; anything that can
change between calls (live availability, today's hours, current prices,
reservations) lives here so it can be queried authoritatively.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable

from .config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS hours (
    day TEXT PRIMARY KEY,         -- mon..sun
    open_time TEXT NOT NULL,      -- 'HH:MM' or '00:00' for 24h
    close_time TEXT NOT NULL,     -- 'HH:MM' or '23:59' for 24h
    is_24h INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing (
    vehicle_type TEXT NOT NULL,   -- car | motorcycle | oversized
    duration TEXT NOT NULL,       -- hourly | daily | weekly | monthly
    zone TEXT NOT NULL,           -- covered | rooftop
    price_usd REAL NOT NULL,
    PRIMARY KEY (vehicle_type, duration, zone)
);

CREATE TABLE IF NOT EXISTS spots (
    zone TEXT PRIMARY KEY,        -- L1 | L2 | L3 | Rooftop
    total INTEGER NOT NULL,
    occupied INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    car_plate TEXT NOT NULL,
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    status TEXT NOT NULL,         -- pending | confirmed | cancelled
    created_at TEXT NOT NULL
);
"""

SEED_HOURS = [
    ("mon", "00:00", "23:59", 1),
    ("tue", "00:00", "23:59", 1),
    ("wed", "00:00", "23:59", 1),
    ("thu", "00:00", "23:59", 1),
    ("fri", "00:00", "23:59", 1),
    ("sat", "00:00", "23:59", 1),
    ("sun", "00:00", "23:59", 1),
]

# Covered zones (L1/L2/L3) share a price; rooftop is discounted.
SEED_PRICING = [
    ("car", "hourly", "covered", 4.50),
    ("car", "daily", "covered", 28.00),
    ("car", "weekly", "covered", 140.00),
    ("car", "monthly", "covered", 380.00),
    ("car", "hourly", "rooftop", 3.00),
    ("car", "daily", "rooftop", 18.00),
    ("car", "weekly", "rooftop", 95.00),
    ("car", "monthly", "rooftop", 240.00),
    ("motorcycle", "hourly", "covered", 2.00),
    ("motorcycle", "daily", "covered", 12.00),
    ("oversized", "hourly", "rooftop", 5.50),
    ("oversized", "daily", "rooftop", 34.00),
]

SEED_SPOTS = [
    ("L1", 130, 92),
    ("L2", 120, 110),  # nearly full because of EV chargers
    ("L3", 100, 41),
    ("Rooftop", 100, 23),
]


@contextmanager
def _conn():
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    con = sqlite3.connect(settings.db_path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(force_reseed: bool = False) -> None:
    with _conn() as con:
        con.executescript(SCHEMA)
        if force_reseed:
            con.execute("DELETE FROM hours")
            con.execute("DELETE FROM pricing")
            con.execute("DELETE FROM spots")
        if con.execute("SELECT COUNT(*) FROM hours").fetchone()[0] == 0:
            con.executemany(
                "INSERT INTO hours VALUES (?, ?, ?, ?)", SEED_HOURS
            )
        if con.execute("SELECT COUNT(*) FROM pricing").fetchone()[0] == 0:
            con.executemany(
                "INSERT INTO pricing VALUES (?, ?, ?, ?)", SEED_PRICING
            )
        if con.execute("SELECT COUNT(*) FROM spots").fetchone()[0] == 0:
            con.executemany(
                "INSERT INTO spots VALUES (?, ?, ?)", SEED_SPOTS
            )


# ---------- read helpers used by chatbot tools ----------

def get_hours(day: str | None = None) -> list[dict]:
    with _conn() as con:
        if day:
            rows = con.execute(
                "SELECT * FROM hours WHERE day = ?", (day.lower()[:3],)
            ).fetchall()
        else:
            rows = con.execute("SELECT * FROM hours").fetchall()
        return [dict(r) for r in rows]


def get_pricing(
    vehicle_type: str | None = None,
    duration: str | None = None,
    zone: str | None = None,
) -> list[dict]:
    sql = "SELECT * FROM pricing WHERE 1=1"
    args: list = []
    if vehicle_type:
        sql += " AND vehicle_type = ?"
        args.append(vehicle_type)
    if duration:
        sql += " AND duration = ?"
        args.append(duration)
    if zone:
        sql += " AND zone = ?"
        args.append(zone)
    with _conn() as con:
        return [dict(r) for r in con.execute(sql, args).fetchall()]


def get_availability(zone: str | None = None) -> list[dict]:
    with _conn() as con:
        if zone:
            rows = con.execute(
                "SELECT * FROM spots WHERE zone = ?", (zone,)
            ).fetchall()
        else:
            rows = con.execute("SELECT * FROM spots").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["available"] = d["total"] - d["occupied"]
            out.append(d)
        return out


def create_booking(
    first_name: str,
    last_name: str,
    car_plate: str,
    start_ts: str,
    end_ts: str,
) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO bookings
                 (first_name, last_name, car_plate, start_ts, end_ts, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
            (
                first_name,
                last_name,
                car_plate,
                start_ts,
                end_ts,
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ),
        )
        return int(cur.lastrowid)


def list_bookings(status: str | None = None) -> list[dict]:
    with _conn() as con:
        if status:
            rows = con.execute(
                "SELECT * FROM bookings WHERE status = ? ORDER BY id DESC",
                (status,),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM bookings ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Initialised DB at {settings.db_path}")
    print("Spots:", get_availability())
