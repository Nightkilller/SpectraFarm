"""
SpectraFarm — FastAPI REST API

Exposes the AgriN Python backend (satellite pipelines, ML classifier,
stress analysis, Gemini AI advisory) as JSON endpoints for the React
frontend to consume.

Run:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Setup paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("AGRIN_MODE", "live")

from src.config.settings import get_settings
from src.data.schemas import DataSource
from src.data.demo_data import (
    generate_ndvi_timeseries,
    generate_sar_observations,
    get_demo_crop_prediction,
    get_demo_farm,
    get_demo_stress_assessment,
    get_demo_farm_analysis,
    get_demo_advisory,
)
from src.features.feature_extraction import (
    combine_features,
    extract_optical_features,
    extract_sar_features,
)
from src.intelligence.stress_analysis import assess_stress
from src.intelligence.farm_analyzer import FarmAnalyzer
from src.ml.crop_classifier import CropClassifierService
from src.ai.gemini_client import generate_advisory, ask_question

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# App Setup
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="SpectraFarm API",
    description="AgriN satellite intelligence backend — NDVI, SAR, ML crop classification, stress analysis, and Gemini AI advisory.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-init singletons
_analyzer: FarmAnalyzer | None = None
_classifier: CropClassifierService | None = None


def get_analyzer() -> FarmAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = FarmAnalyzer()
    return _analyzer


def get_classifier() -> CropClassifierService:
    global _classifier
    if _classifier is None:
        _classifier = CropClassifierService()
    return _classifier


# ═══════════════════════════════════════════════════════════════════════════
# Request / Response Models
# ═══════════════════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    question: str
    farm_id: str = "demo_farm"
    language: str = "en"


class AskResponse(BaseModel):
    answer: str
    farm_id: str
    language: str


class FarmSummary(BaseModel):
    farm_id: str
    name: str
    latitude: float
    longitude: float
    crop: str | None
    area_ha: float | None
    data_source: str


class MetricsResponse(BaseModel):
    farm_id: str
    current_ndvi: float | None
    ndvi_trend: str | None
    stress_level: str | None
    predicted_crop: str | None
    crop_confidence: float | None
    sar_vv_db: float | None
    sar_vh_db: float | None
    health_trend: str | None
    data_source: str
    observation_date: str | None


class AdvisoryResponse(BaseModel):
    farm_id: str
    language: str
    summary: str
    advisory_text: str
    model_version: str
    data_source: str


class TimeSeriesPoint(BaseModel):
    date: str
    ndvi: float | None
    sar_vv_db: float | None
    sar_vh_db: float | None


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "spectrafarm-api", "version": "1.0.0"}


@app.get("/api/farms", response_model=list[FarmSummary])
def list_farms():
    """List available farms."""
    farm = get_demo_farm()
    return [
        FarmSummary(
            farm_id=farm.farm_id,
            name=farm.name,
            latitude=farm.latitude,
            longitude=farm.longitude,
            crop=farm.crop.value if farm.crop else None,
            area_ha=farm.area_ha,
            data_source=farm.data_source.value,
        )
    ]


@app.get("/api/farms/{farm_id}/metrics", response_model=MetricsResponse)
def get_farm_metrics(farm_id: str):
    """Get current satellite metrics and ML predictions for a farm."""
    try:
        analyzer = get_analyzer()
        analysis = analyzer.analyze()

        # Extract SAR data
        sar_obs = [o for o in analysis.recent_observations if o.vv is not None]
        latest_sar = sar_obs[-1] if sar_obs else None

        return MetricsResponse(
            farm_id=analysis.farm.farm_id,
            current_ndvi=analysis.ndvi_current,
            ndvi_trend=analysis.ndvi_trend.value if analysis.ndvi_trend else None,
            stress_level=analysis.stress_assessment.stress_level.value if analysis.stress_assessment else None,
            predicted_crop=analysis.crop_prediction.predicted_crop.value if analysis.crop_prediction else None,
            crop_confidence=analysis.crop_prediction.confidence if analysis.crop_prediction else None,
            sar_vv_db=latest_sar.vv if latest_sar else None,
            sar_vh_db=latest_sar.vh if latest_sar else None,
            health_trend=analysis.ndvi_trend.value if analysis.ndvi_trend else None,
            data_source=analysis.data_source.value,
            observation_date=str(analysis.observation_date) if analysis.observation_date else None,
        )
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/farms/{farm_id}/timeseries", response_model=list[TimeSeriesPoint])
def get_farm_timeseries(farm_id: str, days: int = Query(default=90, ge=7, le=365)):
    """Get NDVI and SAR time series for a farm."""
    try:
        # Generate observations
        ndvi_obs = generate_ndvi_timeseries(farm_id)
        sar_obs = generate_sar_observations(farm_id)

        # Build lookup for SAR by date
        sar_by_date = {o.observation_date: o for o in sar_obs}

        points = []
        for obs in ndvi_obs:
            sar = sar_by_date.get(obs.observation_date)
            points.append(TimeSeriesPoint(
                date=str(obs.observation_date),
                ndvi=obs.ndvi,
                sar_vv_db=sar.vv if sar else None,
                sar_vh_db=sar.vh if sar else None,
            ))

        # Also add SAR-only dates
        ndvi_dates = {o.observation_date for o in ndvi_obs}
        for obs in sar_obs:
            if obs.observation_date not in ndvi_dates:
                points.append(TimeSeriesPoint(
                    date=str(obs.observation_date),
                    ndvi=None,
                    sar_vv_db=obs.vv,
                    sar_vh_db=obs.vh,
                ))

        points.sort(key=lambda p: p.date)
        return points

    except Exception as e:
        logger.error(f"Failed to get timeseries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/farms/{farm_id}/advisory", response_model=AdvisoryResponse)
def get_farm_advisory(farm_id: str, language: str = Query(default="en")):
    """Generate AI advisory for a farm using Gemini."""
    try:
        analyzer = get_analyzer()
        analysis = analyzer.analyze()
        advisory = generate_advisory(analysis, language)

        return AdvisoryResponse(
            farm_id=advisory.farm_id,
            language=advisory.language,
            summary=advisory.observations_summary,
            advisory_text=advisory.advisory_text,
            model_version=advisory.model_version,
            data_source=advisory.data_source.value,
        )
    except Exception as e:
        logger.error(f"Failed to generate advisory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask", response_model=AskResponse)
def ask_agrin(req: AskRequest):
    """Answer a farmer's question using AI with farm context."""
    try:
        analyzer = get_analyzer()
        analysis = analyzer.analyze()
        answer = ask_question(req.question, analysis, req.language)

        return AskResponse(
            answer=answer,
            farm_id=req.farm_id,
            language=req.language,
        )
    except Exception as e:
        logger.error(f"Failed to answer question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Static File Serving (production — serves built React frontend)
# ═══════════════════════════════════════════════════════════════════════════

frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
