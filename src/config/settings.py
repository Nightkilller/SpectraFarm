"""
AgriN — Configuration Loader

Loads application settings from YAML config files and environment variables.
Environment variables (from .env) override YAML defaults where applicable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Project root is two levels up from this file (src/config/settings.py → agriN/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the config directory."""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Config file not found: {filepath}. "
            f"Ensure the config/ directory exists at {CONFIG_DIR}"
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings:
    """
    Central configuration for AgriN.

    Loads from:
    1. config/settings.yaml  — app settings, pilot region, satellite config
    2. config/thresholds.yaml — NDVI / stress / SAR thresholds
    3. config/crops.yaml — crop class definitions
    4. .env — secrets and mode override

    Usage:
        from src.config.settings import get_settings
        settings = get_settings()
        print(settings.mode)           # "demo" or "live"
        print(settings.pilot_region)   # dict with lat/lon/bbox
    """

    def __init__(self) -> None:
        # Load .env (if present) — does NOT override existing env vars
        env_path = PROJECT_ROOT / ".env"
        load_dotenv(dotenv_path=env_path, override=False)

        # Load YAML configs
        self._settings = _load_yaml("settings.yaml")
        self._thresholds = _load_yaml("thresholds.yaml")
        self._crops = _load_yaml("crops.yaml")

    # ── App ──────────────────────────────────────────────────────────────

    @property
    def app_name(self) -> str:
        return self._settings["app"]["name"]

    @property
    def app_version(self) -> str:
        return self._settings["app"]["version"]

    @property
    def mode(self) -> str:
        """Application mode: 'demo' or 'live'. Env var overrides YAML."""
        return os.getenv("AGRIN_MODE", self._settings["app"].get("mode", "demo"))

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    # ── Pilot Region & ML Reference Regions ─────────────────────────────

    @property
    def pilot_region(self) -> dict[str, Any]:
        return self._settings["pilot_region"]

    @property
    def ml_reference_region(self) -> dict[str, Any]:
        return self._settings.get("ml_reference_region", {})

    @property
    def secondary_ml_reference_region(self) -> dict[str, Any]:
        return self._settings.get("secondary_ml_reference_region", {})

    # ── Satellite ────────────────────────────────────────────────────────

    @property
    def sentinel2_config(self) -> dict[str, Any]:
        return self._settings["satellite"]["sentinel2"]

    @property
    def sentinel1_config(self) -> dict[str, Any]:
        return self._settings["satellite"]["sentinel1"]

    @property
    def lookback_months(self) -> int:
        return self._settings["date_range"]["lookback_months"]

    # ── Thresholds ───────────────────────────────────────────────────────

    @property
    def ndvi_thresholds(self) -> dict[str, Any]:
        return self._thresholds["ndvi"]

    @property
    def ndwi_thresholds(self) -> dict[str, Any]:
        return self._thresholds["ndwi"]

    @property
    def stress_thresholds(self) -> dict[str, Any]:
        return self._thresholds["stress"]

    @property
    def sar_thresholds(self) -> dict[str, Any]:
        return self._thresholds["sar"]

    # ── Crops ────────────────────────────────────────────────────────────

    @property
    def crop_classes(self) -> list[dict[str, Any]]:
        return self._crops["classes"]

    @property
    def crop_ids(self) -> list[str]:
        return [c["id"] for c in self.crop_classes]

    @property
    def seasons(self) -> dict[str, Any]:
        return self._crops["seasons"]

    # ── API Keys ─────────────────────────────────────────────────────────

    @property
    def gemini_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or None

    @property
    def gee_project(self) -> str | None:
        return os.getenv("GEE_PROJECT") or None

    # ── Languages ────────────────────────────────────────────────────────

    @property
    def languages(self) -> list[dict[str, str]]:
        return self._settings.get("languages", [{"code": "en", "name": "English"}])

    @property
    def default_language(self) -> str:
        return self._settings.get("default_language", "en")


# ── Singleton accessor ───────────────────────────────────────────────────

_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Get the singleton Settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings_instance
    _settings_instance = None
