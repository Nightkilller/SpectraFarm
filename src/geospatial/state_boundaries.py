"""
AgriN — State Boundary & Masking Helper

Provides exact polygon geometries, bounding boxes, and inverse masking
for Indian states so maps display strictly the selected state.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from shapely.geometry import Point, Polygon, mapping, shape

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEOJSON_PATH = PROJECT_ROOT / "data" / "boundaries" / "india_states.geojson"

_CACHED_GEOJSON: Optional[Dict[str, Any]] = None


def _load_geojson() -> Dict[str, Any]:
    global _CACHED_GEOJSON
    if _CACHED_GEOJSON is not None:
        return _CACHED_GEOJSON

    if not GEOJSON_PATH.exists():
        logger.warning(f"State boundaries file not found at {GEOJSON_PATH}")
        return {"type": "FeatureCollection", "features": []}

    try:
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            _CACHED_GEOJSON = json.load(f)
        return _CACHED_GEOJSON
    except Exception as e:
        logger.error(f"Failed to load state boundaries: {e}")
        return {"type": "FeatureCollection", "features": []}


def get_state_mask_and_bounds(
    lat: float, lon: float
) -> Tuple[str, Dict[str, float], Any, Any]:
    """
    Given (lat, lon), detect which Indian state contains the coordinate.

    Returns:
        state_name: Name of the detected state (e.g. "Madhya Pradesh")
        bounds_dict: {"south": miny, "west": minx, "north": maxy, "east": maxx}
        state_geom_geojson: GeoJSON mapping of the exact state polygon
        mask_geom_geojson: GeoJSON mapping of the inverse world mask (covers everything outside the state)
    """
    data = _load_geojson()
    pt = Point(lon, lat)

    matched_feat = None
    # 1. Direct containment check
    for feat in data.get("features", []):
        try:
            geom = shape(feat["geometry"])
            if geom.contains(pt) or geom.distance(pt) < 0.05:
                matched_feat = feat
                break
        except Exception:
            continue

    # 2. Fallback to closest state if outside exact boundary
    if not matched_feat and data.get("features"):
        min_dist = float("inf")
        for feat in data.get("features", []):
            try:
                geom = shape(feat["geometry"])
                d = geom.distance(pt)
                if d < min_dist:
                    min_dist = d
                    matched_feat = feat
            except Exception:
                continue

    if not matched_feat:
        # Generic fallback
        state_name = "India"
        bounds = {"south": lat - 1.5, "west": lon - 1.5, "north": lat + 1.5, "east": lon + 1.5}
        return state_name, bounds, None, None

    props = matched_feat.get("properties", {})
    state_name = props.get("NAME_1") or props.get("st_nm") or props.get("STATE") or "State"
    state_geom = shape(matched_feat["geometry"])
    minx, miny, maxx, maxy = state_geom.bounds

    # World polygon covering entire globe (lon, lat)
    world_poly = Polygon([(-180, -90), (180, -90), (180, 90), (-180, 90), (-180, -90)])
    try:
        mask_poly = world_poly.difference(state_geom)
        mask_geom_json = mapping(mask_poly)
    except Exception as e:
        logger.warning(f"Could not compute inverse mask: {e}")
        mask_geom_json = None

    bounds = {
        "south": float(miny),
        "west": float(minx),
        "north": float(maxy),
        "east": float(maxx),
    }

    return state_name, bounds, mapping(state_geom), mask_geom_json
