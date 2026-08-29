"""
AgriN — Farm Analyzer (Agricultural Intelligence Layer)

This is the central orchestrator that combines all backend outputs into
a structured FarmAnalysis object.  This is the SINGLE source of truth
that Gemini receives — Gemini never accesses raw satellite data directly.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from src.config.settings import get_settings
from src.data.schemas import (
    DataSource,
    Farm,
    FarmAnalysis,
    HealthTrend,
    SatelliteObservation,
)
from src.data.demo_data import (
    generate_ndvi_timeseries,
    generate_sar_observations,
    get_demo_crop_prediction,
    get_demo_farm,
    get_demo_farm_analysis,
    get_demo_stress_assessment,
)
from src.features.feature_extraction import (
    combine_features,
    extract_optical_features,
    extract_sar_features,
)
from src.geospatial.gee_client import (
    is_gee_available,
    get_dynamic_aoi,
)
from src.geospatial.timeseries import extract_ndvi_timeseries
from src.geospatial.sar import extract_sar_timeseries
from src.intelligence.stress_analysis import assess_stress
from src.ml.crop_classifier import CropClassifierService

logger = logging.getLogger(__name__)


class FarmAnalyzer:
    """
    Orchestrates the full analysis pipeline:

    1. Get satellite observations (live or demo)
    2. Extract features
    3. Run crop classification
    4. Run stress detection
    5. Bundle into FarmAnalysis

    The output is a structured, validated FarmAnalysis object.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.classifier = CropClassifierService()

    def analyze(self, farm: Farm | None = None) -> FarmAnalysis:
        """
        Run the full analysis pipeline for a farm.

        If GEE is unavailable or the classifier is untrained, falls back to
        demo data for those components.
        """
        if farm is None:
            farm = get_demo_farm()

        # Determine mode
        use_live = self.settings.mode == "live" and is_gee_available()

        if use_live:
            try:
                return self._analyze_live(farm)
            except Exception as e:
                logger.warning(f"Live analysis failed ({e}) — falling back to demo mode.")
                return self._analyze_demo(farm)
        else:
            return self._analyze_demo(farm)

    def _analyze_live(self, farm: Farm) -> FarmAnalysis:
        """Run analysis using real satellite data from GEE."""
        logger.info(f"Running LIVE analysis for farm: {farm.name}")

        end_date = date.today()
        start_date = end_date - timedelta(days=30 * self.settings.lookback_months)

        aoi = get_dynamic_aoi(farm.latitude, farm.longitude)
        cloud_max = self.settings.sentinel2_config.get("cloud_cover_max", 20.0)

        # 1. Extract real Sentinel-2 and Sentinel-1 time series
        ndvi_ts = extract_ndvi_timeseries(
            aoi=aoi,
            start_date=start_date,
            end_date=end_date,
            max_cloud_percentage=cloud_max,
        )
        sar_ts = extract_sar_timeseries(
            aoi=aoi,
            start_date=start_date,
            end_date=end_date,
        )

        s2_obs = [
            SatelliteObservation(
                observation_date=pt.observation_date,
                satellite="Sentinel-2",
                farm_id=farm.farm_id,
                ndvi=pt.mean_ndvi,
                cloud_cover=pt.cloud_percentage,
                data_source=DataSource.LIVE,
            )
            for pt in ndvi_ts.points
        ]

        s1_obs = [
            SatelliteObservation(
                observation_date=pt.observation_date,
                satellite="Sentinel-1",
                farm_id=farm.farm_id,
                vv=pt.mean_vv,
                vh=pt.mean_vh,
                vh_vv_ratio=pt.mean_vv_vh_ratio,
                data_source=DataSource.LIVE,
            )
            for pt in sar_ts.points
        ]

        all_observations = s2_obs + s1_obs

        if not s2_obs:
            logger.warning("No Sentinel-2 observations found — falling back to demo")
            return self._analyze_demo(farm)

        # 2. Extract features
        optical_features = extract_optical_features(s2_obs)
        sar_features = extract_sar_features(s1_obs)
        combined = combine_features(optical_features, sar_features)

        # 3. Crop classification
        if self.classifier.is_trained() and combined:
            crop_prediction = self.classifier.predict(combined, farm.farm_id)
        else:
            crop_prediction = get_demo_crop_prediction(farm.farm_id)

        # 4. Stress assessment
        stress = assess_stress(s2_obs, farm.farm_id)

        # 5. Trend
        ndvi_current = s2_obs[-1].ndvi if s2_obs else None
        ndvi_previous = s2_obs[-2].ndvi if len(s2_obs) >= 2 else None
        ndvi_trend = stress.trend

        return FarmAnalysis(
            farm=farm,
            crop_prediction=crop_prediction,
            stress_assessment=stress,
            recent_observations=all_observations,
            ndvi_current=ndvi_current,
            ndvi_previous=ndvi_previous,
            ndvi_trend=ndvi_trend,
            vci_percentage=stress.vci_percentage,
            vci_stress_level=stress.vci_stress_level,
            observation_date=s2_obs[-1].observation_date if s2_obs else None,
            data_source=DataSource.LIVE,
        )

    def _analyze_demo(self, farm: Farm) -> FarmAnalysis:
        """Run analysis using demo data."""
        logger.info(f"Running DEMO analysis for farm: {farm.name}")

        # Generate demo observations
        s2_obs = generate_ndvi_timeseries(farm.farm_id)
        s1_obs = generate_sar_observations(farm.farm_id)

        # Extract features from demo data
        optical_features = extract_optical_features(s2_obs)
        sar_features = extract_sar_features(s1_obs)
        combined = combine_features(optical_features, sar_features)

        # Use classifier if trained, otherwise demo prediction
        if self.classifier.is_trained() and combined:
            crop_prediction = self.classifier.predict(combined, farm.farm_id)
        else:
            crop_prediction = get_demo_crop_prediction(farm.farm_id)

        # Run stress detection on demo observations
        stress = assess_stress(s2_obs, farm.farm_id)

        return FarmAnalysis(
            farm=farm,
            crop_prediction=crop_prediction,
            stress_assessment=stress,
            recent_observations=s2_obs + s1_obs,
            ndvi_current=stress.ndvi_current,
            ndvi_previous=stress.ndvi_previous,
            ndvi_trend=stress.trend,
            vci_percentage=stress.vci_percentage,
            vci_stress_level=stress.vci_stress_level,
            observation_date=s2_obs[-1].observation_date if s2_obs else date.today(),
            data_source=DataSource.DEMO,
        )
