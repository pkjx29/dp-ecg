"""Pydantic request/response schemas for the API."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator

N_LEADS, N_SAMPLES = 12, 1000


class PredictRequest(BaseModel):
    """One preprocessed ECG: 12 leads x 1000 samples (10 s at 100 Hz, band-passed, in mV)."""
    run_id: str = Field(..., max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    waveform: list[list[float]]

    @field_validator("waveform")
    @classmethod
    def check_shape(cls, w: list[list[float]]) -> list[list[float]]:
        if len(w) != N_LEADS or any(len(lead) != N_SAMPLES for lead in w):
            raise ValueError(f"waveform must be {N_LEADS} lists of {N_SAMPLES} numbers")
        if not all(math.isfinite(v) for lead in w for v in lead):
            raise ValueError("waveform contains non-finite values")
        return w


class PredictResponse(BaseModel):
    run_id: str
    score: float
    threshold: float
    note: str = "Experimental model output on a research task. Not a diagnosis."


class Health(BaseModel):
    status: str
    version: str
