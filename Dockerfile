# syntax=docker/dockerfile:1
#
# botalpaca — Professional Telegram + Alpaca trading assistant.
#
# Multi-stage build: the wheel is produced in a builder image and installed into
# a slim runtime that carries no build toolchain. The runtime runs as a
# non-root user and expects secrets ONLY through environment variables.

# ----------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build toolchain is needed only here (pandas/numpy wheels are manylinux, but
# this keeps the build reproducible if a source distribution is ever used).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md alembic.ini ./
COPY botalpaca ./botalpaca
COPY migrations ./migrations

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install .

# ----------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="botalpaca" \
      org.opencontainers.image.description="Professional Telegram trading assistant for Alpaca (paper/live isolated)" \
      org.opencontainers.image.source="https://github.com/Ranielm30/botalpaca"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    TZ=America/New_York \
    # Fly.io mounts its persistent volume here; SQLite lives on the volume so
    # the journal, the audit trail and the protection state survive restarts.
    DATA_DIR=/data \
    DATABASE_URL=sqlite+aiosqlite:////data/botalpaca.db \
    LOG_LEVEL=INFO \
    LOG_JSON=true \
    # PAPER is the startup default. Switching to REAL is a deliberate, confirmed
    # action performed from Telegram with /modo - never an automatic one.
    ACTIVE_TRADING_ENVIRONMENT=PAPER

# tzdata is required for TIMEZONE handling (session VWAP, market clock).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

COPY --from=builder /opt/venv /opt/venv

# Application code and migrations. Migrations must ship in the image because the
# startup sequence runs `alembic upgrade head` against the mounted volume.
WORKDIR /app
COPY --from=builder /build/botalpaca ./botalpaca
COPY --from=builder /build/migrations ./migrations
COPY --from=builder /build/alembic.ini ./alembic.ini

# Pre-create the data directory and hand it to the unprivileged user.
RUN mkdir -p /data \
    && useradd --create-home --uid 10001 botuser \
    && chown -R botuser:botuser /app /data
USER botuser

VOLUME ["/data"]

# `python -m botalpaca --check` starts the app, verifies the database and the
# Alpaca credentials of the active environment, prints health, and exits 0.
# It never sends an order, which makes it a safe container health probe.
HEALTHCHECK --interval=60s --timeout=30s --start-period=45s --retries=3 \
    CMD ["/opt/venv/bin/python", "-m", "botalpaca", "--check"]

CMD ["python", "-m", "botalpaca"]
