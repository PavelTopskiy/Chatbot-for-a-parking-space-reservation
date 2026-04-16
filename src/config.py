"""Central configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "parking-chatbot")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    db_path: str = os.getenv("DB_PATH", str(ROOT / "data" / "dynamic" / "parking.db"))
    static_data_dir: Path = ROOT / "data" / "static"

    top_k: int = int(os.getenv("TOP_K", "4"))

    # SMTP (optional — for email notifications)
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_pass: str = os.getenv("SMTP_PASS", "")
    smtp_from: str = os.getenv("SMTP_FROM", "")
    admin_email: str = os.getenv("ADMIN_EMAIL", "admin@skypark-central.example")

    # MCP server (Stage 3)
    mcp_host: str = os.getenv("MCP_HOST", "0.0.0.0")
    mcp_port: int = int(os.getenv("MCP_PORT", "8001"))
    mcp_url: str = os.getenv("MCP_URL", "http://localhost:8001/mcp")
    mcp_secret: str = os.getenv("MCP_SECRET", "skypark-mcp-secret-change-me")
    reservations_file: str = os.getenv(
        "RESERVATIONS_FILE",
        str(ROOT / "data" / "confirmed_reservations.txt"),
    )


settings = Settings()
