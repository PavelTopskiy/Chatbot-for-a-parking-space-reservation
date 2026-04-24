# ---- builder ----
# Installs dependencies into a wheel cache so the final image is leaner and
# the build cache only invalidates when requirements.txt changes.
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build-essential is needed for any sdists without wheels on arm64.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip wheel --wheel-dir=/wheels -r requirements.txt


# ---- runtime ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Create a non-root user so the containers don't run as root.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

# Install prebuilt wheels from the builder stage.
COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy the application last so code changes don't bust the dep-install layer.
COPY src ./src
COPY eval ./eval
COPY scripts ./scripts
COPY tests ./tests
# Static knowledge-base markdown for RAG ingestion.
COPY data/static ./data/static

# Runtime writable data directory. At deploy time this should be a volume
# so SQLite + the MCP output file survive container restarts.
RUN mkdir -p data/dynamic \
    && chown -R app:app /app

USER app

# Dashboard port by default; override for the MCP server service.
EXPOSE 8000 8001

# Start the user/admin dashboard by default; docker-compose overrides the
# command for the MCP server and (optionally) the orchestrator CLI.
CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
