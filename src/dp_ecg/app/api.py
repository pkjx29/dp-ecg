"""FastAPI service over SAVED results. It never trains or downloads data.

Endpoints: /health, /metadata, /experiments, /privacy-results, and
POST /predict when a checkpoint for the requested run is present.
Responses never include file paths, environment variables or tracebacks.
"""

from __future__ import annotations

import json
from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import artifact_dir
from .schemas import Health, PredictRequest, PredictResponse

MAX_BODY_BYTES = 2_000_000  # a 12 x 1000 float payload as JSON is ~250 kB
DISCLAIMER = "Research demonstration. Not a medical device; outputs are not diagnoses."


def _read_json(kind: str, name: str):
    path = artifact_dir(kind) / name
    return json.loads(path.read_text()) if path.exists() else None


def _runs() -> list[dict]:
    out = []
    for p in sorted((artifact_dir("metrics") / "runs").glob("*.json")):
        if p.name.endswith(".error.json"):
            continue
        r = json.loads(p.read_text())
        out.append({"run_id": r["run_id"], "kind": r["kind"], "status": r["status"],
                    "privacy": r["privacy"], "test": r["test"], "best_epoch": r["best_epoch"]})
    return out


@lru_cache(maxsize=8)
def _load_model(run_id: str):
    from ..models.cnn import build_model
    path = artifact_dir("checkpoints") / f"{run_id}.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def create_app(with_ui: bool = False) -> FastAPI:
    app = FastAPI(title="DP-ECG research API", version=__version__, description=DISCLAIMER)

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if int(request.headers.get("content-length", 0)) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "request too large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(Exception)
    async def hide_internals(request: Request, exc: Exception):
        return JSONResponse({"detail": "internal error"}, status_code=500)

    @app.get("/health", response_model=Health)
    def health():
        return Health(status="ok", version=__version__)

    @app.get("/metadata")
    def metadata():
        return {"project": "Differentially private ECG classification", "task": "PTB-XL NORM vs MI",
                "privacy_unit": "one ECG record (record-level, not patient-level)",
                "data_report": _read_json("reports", "data_report.json"), "disclaimer": DISCLAIMER}

    @app.get("/experiments")
    def experiments():
        runs = _runs()
        return {"status": "ok" if runs else "pending: no completed runs found", "runs": runs}

    @app.get("/privacy-results")
    def privacy_results():
        return {"summary_statistics": _read_json("reports", "dp_summary_statistics.json"),
                "mechanism_verification": _read_json("reports", "mechanism_verification.json"),
                "membership_inference": (_read_json("metrics", "membership_inference.json") or {}).get("results"),
                "status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        run = next((r for r in _runs() if r["run_id"] == req.run_id), None)
        model = _load_model(req.run_id) if run else None
        if model is None:
            raise HTTPException(404, "no model checkpoint available for this run_id")
        x = torch.tensor([req.waveform], dtype=torch.float32)
        with torch.no_grad():
            score = float(torch.sigmoid(model(x))[0])
        return PredictResponse(run_id=req.run_id, score=score, threshold=run["test"]["threshold"])

    if with_ui:
        import gradio as gr

        from .ui import build_ui
        app = gr.mount_gradio_app(app, build_ui(), path="/")
    return app
