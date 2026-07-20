# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Cloud Run serving image for the credit default risk API.
#
# Multi-stage: dependencies are compiled in a builder stage and only the
# resulting virtualenv is copied forward. The final image carries no compiler
# toolchain, which cuts both size and attack surface.
#
# Python 3.12 rather than 3.14: at build time 3.12 has the broadest binary wheel
# coverage across this dependency set, so nothing needs compiling from source.
# The application code is version-agnostic (requires-python = ">=3.11").
#
# Build from the repository root, not from docker/:
#     docker build -f docker/Dockerfile -t credit-risk-api .
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.12

# --------------------------- builder ---------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed only if a wheel is missing for this platform; it
# stays in the builder and never reaches the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-serving.txt .

# CPU-only PyTorch. The default PyPI wheel pulls in several GB of CUDA
# libraries that are useless on Cloud Run, where there is no GPU, and that
# inflate cold starts.
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-serving.txt

# --------------------------- runtime ---------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=8080

# Run as an unprivileged user. Cloud Run does not require root, and a
# compromised process should not own the filesystem.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Copied individually so the image contains only what serving needs - no data,
# no notebooks, no terraform state.
COPY --chown=appuser:appuser src/ ./src/
# sql/ is not optional: the API executes these exact files at request time to
# compute features, which is what keeps training and serving in lockstep.
COPY --chown=appuser:appuser sql/ ./sql/
COPY --chown=appuser:appuser config/ ./config/

# Model artifacts. Present when the image is built after a training run; when
# absent the service starts and reports "degraded" on /health rather than
# crash-looping, and can instead load from Cloud Storage at startup.
COPY --chown=appuser:appuser data/models/ ./data/models/

USER appuser

EXPOSE 8080

# Cloud Run has its own health probing; this helps `docker run` locally and any
# other orchestrator.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8080\")}/health', timeout=4).status==200 else 1)"

# Shell form so ${PORT} is expanded at runtime - Cloud Run injects the port and
# will not route traffic to a container listening anywhere else.
#
# One worker per container by design: Cloud Run scales by adding instances, and
# extra in-container workers would multiply the model's memory footprint against
# a fixed per-instance limit.
CMD exec uvicorn credit_risk.serving.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --log-config /dev/null
