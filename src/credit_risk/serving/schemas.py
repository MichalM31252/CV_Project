"""Request and response schemas for the prediction API.

The API deliberately accepts **raw account data** - the same fields a core
banking system holds - rather than engineered features.

Requiring callers to submit pre-computed features would mean every caller
reimplements ``sql/02_features.sql``. The moment one of them rounds a ratio
differently, predictions diverge from training with nothing to indicate it. That
failure mode - training/serving skew - is among the most common ways a working
model quietly degrades in production.

Validation bounds are drawn from the source codebook, so malformed input is
rejected at the edge with a precise message rather than silently scored.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# Repayment status scale from the codebook:
#   -2 no consumption, -1 paid in full, 0 revolving credit, 1..9 months delayed.
PayStatus = Annotated[int, Field(ge=-2, le=9)]
# Monetary amounts. Bills may be negative (overpayment leaves a credit balance).
Money = Annotated[float, Field(ge=-1_000_000, le=100_000_000)]
NonNegativeMoney = Annotated[float, Field(ge=0, le=100_000_000)]


class ClientRecord(BaseModel):
    """One credit-card account with six months of history.

    Month index 1 is the most recent month, 6 the oldest - matching the warehouse
    convention rather than the source file's ``PAY_0`` quirk.
    """

    client_id: int = Field(..., ge=0, description="Account identifier, used for prediction logging")
    limit_bal: Annotated[float, Field(gt=0, le=100_000_000)] = Field(
        ..., description="Credit limit in NT$"
    )
    sex: Literal[1, 2] = Field(
        1,
        description="1=male, 2=female. Excluded from the model; recorded for fairness auditing only",
    )
    education: Annotated[int, Field(ge=0, le=6)] = Field(
        ...,
        description="1=graduate school, 2=university, 3=high school, other values map to 'other'",
    )
    marriage: Annotated[int, Field(ge=0, le=3)] = Field(
        ..., description="1=married, 2=single, 3/0=other"
    )
    age: Annotated[int, Field(ge=18, le=100)] = Field(..., description="Age in years")

    pay_status_1: PayStatus = Field(..., description="Repayment status, most recent month")
    pay_status_2: PayStatus
    pay_status_3: PayStatus
    pay_status_4: PayStatus
    pay_status_5: PayStatus
    pay_status_6: PayStatus = Field(..., description="Repayment status, oldest month")

    bill_amt_1: Money = Field(..., description="Statement balance, most recent month")
    bill_amt_2: Money
    bill_amt_3: Money
    bill_amt_4: Money
    bill_amt_5: Money
    bill_amt_6: Money

    pay_amt_1: NonNegativeMoney = Field(..., description="Amount paid during the most recent month")
    pay_amt_2: NonNegativeMoney
    pay_amt_3: NonNegativeMoney
    pay_amt_4: NonNegativeMoney
    pay_amt_5: NonNegativeMoney
    pay_amt_6: NonNegativeMoney

    model_config = {
        "json_schema_extra": {
            "example": {
                "client_id": 1,
                "limit_bal": 20000,
                "sex": 2,
                "education": 2,
                "marriage": 1,
                "age": 24,
                "pay_status_1": 2,
                "pay_status_2": 2,
                "pay_status_3": -1,
                "pay_status_4": -1,
                "pay_status_5": -2,
                "pay_status_6": -2,
                "bill_amt_1": 3913,
                "bill_amt_2": 3102,
                "bill_amt_3": 689,
                "bill_amt_4": 0,
                "bill_amt_5": 0,
                "bill_amt_6": 0,
                "pay_amt_1": 0,
                "pay_amt_2": 689,
                "pay_amt_3": 0,
                "pay_amt_4": 0,
                "pay_amt_5": 0,
                "pay_amt_6": 0,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    records: list[ClientRecord] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> BatchPredictionRequest:
        """Reject duplicate ids so responses can be matched back unambiguously."""
        ids = [r.client_id for r in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError("client_id values must be unique within a batch")
        return self


class PredictionResponse(BaseModel):
    client_id: int
    default_probability: float = Field(
        ..., ge=0, le=1, description="Calibrated probability of default next month"
    )
    decision: Literal["flag", "no_flag"] = Field(
        ..., description="Cost-optimal action at the model's operating threshold"
    )
    threshold: float = Field(..., description="Operating threshold applied")
    risk_band: Literal["low", "medium", "high"] = Field(
        ..., description="Coarse band for downstream routing"
    )
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    backend: str


class ModelInfoResponse(BaseModel):
    """Full model card, so a caller can see exactly what is scoring them."""

    model_name: str
    flavor: str
    version: str
    trained_at: str
    git_commit: str | None
    threshold: float
    n_features: int
    feature_names: list[str]
    metrics: dict
    cost_matrix: dict
    libraries: dict
