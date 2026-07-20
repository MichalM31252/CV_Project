"""FastAPI application served on Cloud Run.

Operational behaviour that Cloud Run specifically requires:

* The model is loaded once during startup, not per request. Cloud Run may run
  many concurrent requests per instance; loading in a handler would multiply
  memory and add seconds of latency to unlucky callers.
* Startup also runs one warm-up prediction. The first call through scikit-learn
  and DuckDB costs ~1.3s of lazy initialisation, so without warming, the first
  real user after every cold start pays it.
* ``/health`` is liveness only and never touches the model, so a slow model can
  never cause the platform to kill a healthy container. ``/ready`` reports
  whether the model actually loaded.
* ``X-Cloud-Trace-Context`` is propagated into logs so every line emitted while
  serving a request is grouped in Cloud Logging.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..logging_config import configure_from_settings
from ..monitoring.prediction_log import PredictionLogger
from .predictor import Predictor
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ClientRecord,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# Process-wide state, populated during the startup phase of the lifespan.
_state: dict[str, Any] = {"predictor": None, "prediction_logger": None, "error": None}

# Lightweight in-process counters exposed on /metrics. Deliberately not a
# Prometheus client: Cloud Run scales to zero and instances are ephemeral, so
# scrape-based collection does not fit. Cloud Monitoring reads the structured
# logs instead; these counters are for quick human inspection of one instance.
_metrics: dict[str, float] = defaultdict(float)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_from_settings(settings)
    logger.info(
        "starting service",
        extra={"backend": settings.backend, "flavor": settings.serving.model_flavor},
    )

    try:
        predictor = Predictor(settings)
        # Warm the lazy paths (DuckDB planner, scikit-learn predict) so the first
        # real request does not absorb cold-start cost.
        predictor.predict([_WARMUP_RECORD])
        _state["predictor"] = predictor
        _state["prediction_logger"] = PredictionLogger(settings)
        logger.info("service ready", extra={"model_version": predictor.version})
    except Exception as exc:  # noqa: BLE001 - must surface, not crash the container
        # Starting without a model lets /health and /ready report the fault
        # clearly. A container that exits instead just crash-loops with the real
        # cause buried in restart noise.
        _state["error"] = str(exc)
        logger.exception("failed to load model during startup")

    yield

    predictor = _state.get("predictor")
    if predictor is not None:
        predictor.close()
    prediction_logger = _state.get("prediction_logger")
    if prediction_logger is not None:
        prediction_logger.close()
    logger.info("service stopped")


app = FastAPI(
    title="Credit Default Risk API",
    description=(
        "Calibrated probability that a credit-card client defaults next month, "
        "with a cost-optimal flag decision. Accepts raw account data; feature "
        "engineering runs server-side using the same SQL as the training pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_WARMUP_RECORD: dict[str, Any] = {
    "client_id": 0,
    "limit_bal": 50000.0,
    "sex": 1,
    "education": 2,
    "marriage": 1,
    "age": 35,
    "pay_status_1": 0,
    "pay_status_2": 0,
    "pay_status_3": 0,
    "pay_status_4": 0,
    "pay_status_5": 0,
    "pay_status_6": 0,
    "bill_amt_1": 1000.0,
    "bill_amt_2": 1000.0,
    "bill_amt_3": 1000.0,
    "bill_amt_4": 1000.0,
    "bill_amt_5": 1000.0,
    "bill_amt_6": 1000.0,
    "pay_amt_1": 500.0,
    "pay_amt_2": 500.0,
    "pay_amt_3": 500.0,
    "pay_amt_4": 500.0,
    "pay_amt_5": 500.0,
    "pay_amt_6": 500.0,
}


def _get_predictor() -> Predictor:
    predictor = _state.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"model unavailable: {_state.get('error', 'not loaded')}",
        )
    return predictor


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Time every request and attach Cloud Run's trace id to its logs."""
    start = time.perf_counter()
    trace_header = request.headers.get("X-Cloud-Trace-Context")
    try:
        response = await call_next(request)
    except Exception:
        _metrics["requests_failed"] += 1
        logger.exception(
            "unhandled error",
            extra={"path": request.url.path, "trace_header": trace_header},
        )
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    _metrics["requests_total"] += 1
    _metrics["latency_ms_sum"] += duration_ms

    # Health checks fire constantly; logging them would bury the real traffic.
    if request.url.path not in ("/health", "/ready", "/metrics"):
        logger.info(
            "request handled",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "latency_ms": round(duration_ms, 2),
                "trace_header": trace_header,
            },
        )
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness. Intentionally does not exercise the model."""
    predictor = _state.get("predictor")
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_version=predictor.version if predictor else None,
        backend=get_settings().backend,
    )


@app.get("/ready", tags=["ops"])
async def ready() -> JSONResponse:
    """Readiness. Returns 503 until the model is loaded."""
    if _state.get("predictor") is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "error": _state.get("error")},
        )
    return JSONResponse(content={"ready": True})


@app.get("/metrics", tags=["ops"])
async def metrics() -> dict[str, float]:
    """In-process counters for the current instance."""
    total = _metrics.get("requests_total", 0.0)
    return {
        "requests_total": total,
        "requests_failed": _metrics.get("requests_failed", 0.0),
        "predictions_total": _metrics.get("predictions_total", 0.0),
        "flagged_total": _metrics.get("flagged_total", 0.0),
        "avg_latency_ms": round(_metrics.get("latency_ms_sum", 0.0) / total, 2) if total else 0.0,
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
async def model_info() -> ModelInfoResponse:
    """Model card: version, lineage, threshold, metrics and cost matrix."""
    predictor = _get_predictor()
    metadata = predictor.metadata
    return ModelInfoResponse(
        model_name=metadata["model_name"],
        flavor=metadata["flavor"],
        version=metadata["version"],
        trained_at=metadata["trained_at"],
        git_commit=metadata.get("git_commit"),
        threshold=metadata["threshold"],
        n_features=metadata["n_features"],
        feature_names=metadata["feature_names"],
        metrics=metadata.get("metrics", {}),
        cost_matrix=metadata.get("cost_matrix", {}),
        libraries=metadata.get("libraries", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
async def predict(record: ClientRecord) -> PredictionResponse:
    """Score a single client."""
    predictor = _get_predictor()
    try:
        result = predictor.predict([record.model_dump()])[0]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    _metrics["predictions_total"] += 1
    _metrics["flagged_total"] += result["decision"] == "flag"
    _log_predictions([record.model_dump()], [result])
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["model"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score up to ``serving.max_batch_size`` clients in one call."""
    settings = get_settings()
    if len(request.records) > settings.serving.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"batch of {len(request.records)} exceeds max_batch_size "
                f"{settings.serving.max_batch_size}"
            ),
        )

    predictor = _get_predictor()
    start = time.perf_counter()
    records = [r.model_dump() for r in request.records]
    try:
        results = predictor.predict(records)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    latency_ms = (time.perf_counter() - start) * 1000

    _metrics["predictions_total"] += len(results)
    _metrics["flagged_total"] += sum(r["decision"] == "flag" for r in results)
    _log_predictions(records, results)

    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r) for r in results],
        model_version=predictor.version,
        latency_ms=round(latency_ms, 2),
    )


def _log_predictions(records: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    """Persist predictions for monitoring. Never fails the request."""
    prediction_logger = _state.get("prediction_logger")
    if prediction_logger is None:
        return
    try:
        prediction_logger.log(records, results)
    except Exception:  # noqa: BLE001 - monitoring must not break serving
        logger.exception("failed to log predictions")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    # Cloud Run injects PORT and expects the container to listen on it.
    uvicorn.run(
        "credit_risk.serving.app:app",
        host="0.0.0.0",  # noqa: S104 - required inside a container
        port=int(os.environ.get("PORT", "8080")),
        log_config=None,  # our JSON handler owns formatting
    )
