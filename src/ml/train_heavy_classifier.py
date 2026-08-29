"""
SpectraFarm — Multi-State 75,000+ Sample Ground-Truth Dataset Generator & Model Retraining

Covers key agricultural hubs across 4 major Indian states:
  - Madhya Pradesh (Sehore, Bhopal, Hoshangabad, Ujjain, Vidisha, Dewas)
  - Rajasthan (Sri Ganganagar, Kota, Alwar, Bharatpur, Hanumangarh)
  - Uttar Pradesh (Barabanki, Kannauj, Sitapur, Mathura, Gorakhpur, Muzaffarnagar)
  - Bihar (Samastipur, Patna, Muzaffarpur)

10 Major Crop Classes (75,000 Total Observations):
  1. Wheat (Rabi) - 8,500
  2. Rice (Kharif) - 8,000
  3. Soybean (Kharif - MP & Kota) - 8,000
  4. Mustard (Rabi - Rajasthan & UP) - 8,000
  5. Cotton (Kharif - Rajasthan) - 7,000
  6. Sugarcane (Annual - UP & Bihar) - 7,500
  7. Potato (Rabi - UP) - 7,000
  8. Lentil / Pulses (Rabi - MP & UP) - 7,000
  9. Maize (Kharif/Rabi - Bihar & MP) - 7,000
 10. Gram / Chickpea (Rabi - MP & Rajasthan) - 7,000
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("/Users/adityagupta/Desktop/EPIC project/agriN/models/crop_classifier")
DATA_DIR = Path("/Users/adityagupta/Desktop/EPIC project/agriN/data/ground_truth")

CROP_PROFILES = {
    "Wheat": {
        "n_samples": 8500,
        "ndvi_mean": (0.54, 0.065),
        "ndvi_min": (0.18, 0.04),
        "ndvi_max": (0.78, 0.05),
        "ndvi_std": (0.18, 0.03),
        "ndvi_range": (0.60, 0.06),
        "ndvi_slope": (0.015, 0.005),
        "ndwi_mean": (-0.18, 0.05),
        "blue_mean": (0.09, 0.015),
        "green_mean": (0.12, 0.015),
        "red_mean": (0.11, 0.02),
        "nir_mean": (0.30, 0.04),
        "swir1_mean": (0.21, 0.03),
        "vv_mean": (-10.5, 1.4),
        "vh_mean": (-17.2, 1.5),
        "vv_std": (2.4, 0.4),
        "vh_std": (3.1, 0.5),
        "vh_vv_ratio": (0.21, 0.03),
        "lat_range": (22.5, 30.0),
        "lon_range": (73.5, 85.5),
        "states": ["MP", "UP", "Rajasthan", "Bihar"],
    },
    "Rice": {
        "n_samples": 8000,
        "ndvi_mean": (0.64, 0.07),
        "ndvi_min": (0.10, 0.035),
        "ndvi_max": (0.84, 0.04),
        "ndvi_std": (0.22, 0.04),
        "ndvi_range": (0.74, 0.05),
        "ndvi_slope": (0.022, 0.006),
        "ndwi_mean": (0.12, 0.08),
        "blue_mean": (0.07, 0.012),
        "green_mean": (0.09, 0.015),
        "red_mean": (0.08, 0.015),
        "nir_mean": (0.36, 0.05),
        "swir1_mean": (0.15, 0.03),
        "vv_mean": (-13.8, 1.8),
        "vh_mean": (-20.5, 1.9),
        "vv_std": (3.5, 0.6),
        "vh_std": (4.2, 0.6),
        "vh_vv_ratio": (0.22, 0.04),
        "lat_range": (24.0, 28.5),
        "lon_range": (80.0, 87.5),
        "states": ["UP", "Bihar", "MP"],
    },
    "Soybean": {
        "n_samples": 8000,
        "ndvi_mean": (0.62, 0.06),
        "ndvi_min": (0.14, 0.03),
        "ndvi_max": (0.82, 0.045),
        "ndvi_std": (0.21, 0.035),
        "ndvi_range": (0.68, 0.05),
        "ndvi_slope": (0.020, 0.005),
        "ndwi_mean": (0.02, 0.05),
        "blue_mean": (0.08, 0.012),
        "green_mean": (0.11, 0.014),
        "red_mean": (0.08, 0.014),
        "nir_mean": (0.35, 0.04),
        "swir1_mean": (0.16, 0.025),
        "vv_mean": (-10.2, 1.3),
        "vh_mean": (-16.5, 1.4),
        "vv_std": (2.5, 0.4),
        "vh_std": (3.2, 0.4),
        "vh_vv_ratio": (0.23, 0.03),
        "lat_range": (22.0, 25.5),
        "lon_range": (75.0, 78.5),
        "states": ["MP", "Rajasthan"],
    },
    "Mustard": {
        "n_samples": 8000,
        "ndvi_mean": (0.46, 0.055),
        "ndvi_min": (0.20, 0.04),
        "ndvi_max": (0.68, 0.05),
        "ndvi_std": (0.14, 0.03),
        "ndvi_range": (0.48, 0.05),
        "ndvi_slope": (0.009, 0.004),
        "ndwi_mean": (-0.26, 0.04),
        "blue_mean": (0.12, 0.018),
        "green_mean": (0.15, 0.02),
        "red_mean": (0.13, 0.02),
        "nir_mean": (0.27, 0.03),
        "swir1_mean": (0.24, 0.03),
        "vv_mean": (-9.2, 1.2),
        "vh_mean": (-15.8, 1.4),
        "vv_std": (2.1, 0.3),
        "vh_std": (2.8, 0.4),
        "vh_vv_ratio": (0.23, 0.03),
        "lat_range": (26.0, 30.0),
        "lon_range": (73.0, 79.5),
        "states": ["Rajasthan", "UP", "MP"],
    },
    "Cotton": {
        "n_samples": 7000,
        "ndvi_mean": (0.50, 0.06),
        "ndvi_min": (0.16, 0.03),
        "ndvi_max": (0.72, 0.05),
        "ndvi_std": (0.17, 0.03),
        "ndvi_range": (0.56, 0.05),
        "ndvi_slope": (0.012, 0.004),
        "ndwi_mean": (-0.15, 0.04),
        "blue_mean": (0.10, 0.015),
        "green_mean": (0.13, 0.016),
        "red_mean": (0.11, 0.016),
        "nir_mean": (0.31, 0.035),
        "swir1_mean": (0.22, 0.03),
        "vv_mean": (-8.5, 1.2),
        "vh_mean": (-14.5, 1.3),
        "vv_std": (2.2, 0.3),
        "vh_std": (2.9, 0.4),
        "vh_vv_ratio": (0.25, 0.03),
        "lat_range": (28.5, 30.2),
        "lon_range": (73.5, 75.5),
        "states": ["Rajasthan"],
    },
    "Sugarcane": {
        "n_samples": 7500,
        "ndvi_mean": (0.68, 0.05),
        "ndvi_min": (0.35, 0.05),
        "ndvi_max": (0.86, 0.04),
        "ndvi_std": (0.12, 0.025),
        "ndvi_range": (0.51, 0.05),
        "ndvi_slope": (0.005, 0.003),
        "ndwi_mean": (-0.05, 0.04),
        "blue_mean": (0.08, 0.012),
        "green_mean": (0.10, 0.014),
        "red_mean": (0.07, 0.012),
        "nir_mean": (0.38, 0.04),
        "swir1_mean": (0.16, 0.025),
        "vv_mean": (-7.8, 1.3),
        "vh_mean": (-13.2, 1.4),
        "vv_std": (1.8, 0.3),
        "vh_std": (2.3, 0.3),
        "vh_vv_ratio": (0.29, 0.04),
        "lat_range": (26.0, 30.0),
        "lon_range": (77.0, 84.5),
        "states": ["UP", "Bihar"],
    },
    "Potato": {
        "n_samples": 7000,
        "ndvi_mean": (0.58, 0.06),
        "ndvi_min": (0.15, 0.03),
        "ndvi_max": (0.81, 0.04),
        "ndvi_std": (0.21, 0.03),
        "ndvi_range": (0.66, 0.05),
        "ndvi_slope": (0.024, 0.005),
        "ndwi_mean": (-0.12, 0.05),
        "blue_mean": (0.08, 0.014),
        "green_mean": (0.11, 0.015),
        "red_mean": (0.09, 0.015),
        "nir_mean": (0.33, 0.04),
        "swir1_mean": (0.18, 0.03),
        "vv_mean": (-11.2, 1.3),
        "vh_mean": (-18.0, 1.5),
        "vv_std": (2.6, 0.4),
        "vh_std": (3.2, 0.4),
        "vh_vv_ratio": (0.21, 0.03),
        "lat_range": (26.0, 28.5),
        "lon_range": (78.5, 82.0),
        "states": ["UP", "MP"],
    },
    "Lentil": {
        "n_samples": 7000,
        "ndvi_mean": (0.42, 0.05),
        "ndvi_min": (0.18, 0.04),
        "ndvi_max": (0.62, 0.05),
        "ndvi_std": (0.13, 0.03),
        "ndvi_range": (0.44, 0.05),
        "ndvi_slope": (0.008, 0.003),
        "ndwi_mean": (-0.28, 0.04),
        "blue_mean": (0.11, 0.015),
        "green_mean": (0.13, 0.016),
        "red_mean": (0.13, 0.018),
        "nir_mean": (0.25, 0.03),
        "swir1_mean": (0.25, 0.03),
        "vv_mean": (-12.0, 1.4),
        "vh_mean": (-19.2, 1.5),
        "vv_std": (2.0, 0.3),
        "vh_std": (2.5, 0.4),
        "vh_vv_ratio": (0.19, 0.03),
        "lat_range": (23.0, 26.8),
        "lon_range": (77.0, 85.5),
        "states": ["MP", "UP", "Bihar"],
    },
    "Maize": {
        "n_samples": 7000,
        "ndvi_mean": (0.61, 0.06),
        "ndvi_min": (0.15, 0.04),
        "ndvi_max": (0.83, 0.05),
        "ndvi_std": (0.20, 0.03),
        "ndvi_range": (0.68, 0.05),
        "ndvi_slope": (0.019, 0.005),
        "ndwi_mean": (-0.08, 0.05),
        "blue_mean": (0.08, 0.012),
        "green_mean": (0.11, 0.015),
        "red_mean": (0.09, 0.015),
        "nir_mean": (0.35, 0.04),
        "swir1_mean": (0.17, 0.03),
        "vv_mean": (-8.8, 1.3),
        "vh_mean": (-14.9, 1.4),
        "vv_std": (2.3, 0.4),
        "vh_std": (3.0, 0.4),
        "vh_vv_ratio": (0.25, 0.03),
        "lat_range": (22.5, 27.2),
        "lon_range": (76.0, 87.5),
        "states": ["Bihar", "MP", "Rajasthan"],
    },
    "Gram": {
        "n_samples": 7000,
        "ndvi_mean": (0.45, 0.05),
        "ndvi_min": (0.16, 0.03),
        "ndvi_max": (0.65, 0.05),
        "ndvi_std": (0.15, 0.03),
        "ndvi_range": (0.49, 0.05),
        "ndvi_slope": (0.011, 0.004),
        "ndwi_mean": (-0.25, 0.04),
        "blue_mean": (0.10, 0.014),
        "green_mean": (0.13, 0.016),
        "red_mean": (0.12, 0.018),
        "nir_mean": (0.27, 0.035),
        "swir1_mean": (0.23, 0.03),
        "vv_mean": (-10.8, 1.3),
        "vh_mean": (-17.5, 1.4),
        "vv_std": (2.2, 0.3),
        "vh_std": (2.7, 0.4),
        "vh_vv_ratio": (0.21, 0.03),
        "lat_range": (22.5, 27.5),
        "lon_range": (74.5, 83.5),
        "states": ["MP", "Rajasthan", "UP"],
    },
}

FEATURE_COLS = [
    "blue_mean", "green_mean", "red_mean", "nir_mean", "swir1_mean",
    "ndvi_mean", "ndvi_min", "ndvi_max", "ndvi_std", "ndvi_range", "ndvi_slope",
    "ndwi_mean",
    "vv_mean", "vv_std", "vh_mean", "vh_std", "vh_vv_ratio"
]


def generate_75k_dataset() -> pd.DataFrame:
    """Vectorized high-speed generation of 75,000 calibrated field records."""
    np.random.seed(42)
    logger.info("Generating 75,000+ multi-state agricultural field records...")
    t0 = time.time()
    
    dfs = []
    field_counter = 1

    for crop_name, profile in CROP_PROFILES.items():
        n = profile["n_samples"]
        states = np.random.choice(profile["states"], size=n)
        data = {
            "field_id": [f"FIELD_{i:06d}" for i in range(field_counter, field_counter + n)],
            "crop": [crop_name] * n,
            "state": states,
            "lat": np.round(np.random.uniform(profile["lat_range"][0], profile["lat_range"][1], size=n), 4),
            "lon": np.round(np.random.uniform(profile["lon_range"][0], profile["lon_range"][1], size=n), 4),
        }
        for feat in FEATURE_COLS:
            mu, sigma = profile[feat]
            vals = np.random.normal(mu, sigma, size=n)
            if "ndvi" in feat or "ndwi" in feat:
                vals = np.clip(vals, -1.0, 1.0)
            elif "_mean" in feat and feat not in ("vv_mean", "vh_mean", "ndvi_mean", "ndwi_mean"):
                vals = np.clip(vals, 0.01, 0.99)
            data[feat] = np.round(vals, 6)

        dfs.append(pd.DataFrame(data))
        field_counter += n

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    logger.info(f"Generated {len(full_df):,} field records in {time.time()-t0:.2f}s across {len(CROP_PROFILES)} classes.")
    return full_df


from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score
import joblib

from src.data.load_ground_truth import load_merged_ground_truth, FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "crop_classifier"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ground_truth"


def train_and_save_model(df: pd.DataFrame):
    """Train Random Forest model on the merged real + synthetic dataset."""
    logger.info(f"Preparing dataset: {len(df):,} total samples across {df['crop'].nunique()} crop classes.")
    if "source" in df.columns:
        logger.info("Data source breakdown:\n" + str(df["source"].value_counts()))

    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df["crop"])

    logger.info(f"Class labels mapped ({len(le.classes_)} classes): {dict(enumerate(le.classes_))}")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training on {len(X_train):,} samples, evaluating on {len(X_test):,} holdout test samples...")
    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    logger.info(f"Training completed in {train_time:.2f}s.")

    # Train Accuracy
    y_train_pred = rf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Test Accuracy & Metrics
    y_test_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    kappa = cohen_kappa_score(y_test, y_test_pred)

    logger.info(f"\n{'='*60}")
    logger.info(f"MODEL PERFORMANCE EVALUATION")
    logger.info(f"{'='*60}")
    logger.info(f"Train Set Accuracy ({len(X_train):,} samples): {train_acc*100:.2f}%")
    logger.info(f"Test Set Accuracy  ({len(X_test):,} samples): {test_acc*100:.2f}%")
    logger.info(f"Cohen's Kappa (κ):                    {kappa:.4f}")
    logger.info(f"{'='*60}")
    logger.info("\nPer-Class Classification Report:\n" + classification_report(y_test, y_test_pred, target_names=le.classes_, digits=4))

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "training_features.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved dataset ({csv_path.stat().st_size / (1024*1024):.2f} MB) to {csv_path}")

    model_path = MODELS_DIR / "random_forest.joblib"
    feat_path = MODELS_DIR / "feature_names.joblib"
    le_path = MODELS_DIR / "label_encoder.joblib"

    joblib.dump(rf, model_path)
    joblib.dump(FEATURE_COLS, feat_path)
    joblib.dump(le, le_path)

    logger.info(f"Saved trained model to {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
    logger.info(f"Saved feature names to {feat_path}")
    logger.info(f"Saved label encoder to {le_path}")


if __name__ == "__main__":
    # Check if dataset already exists or generate
    dataset_file = DATA_DIR / "training_features.csv"
    if not dataset_file.exists():
        logger.info("Generating synthetic baseline dataset...")
        df_base = generate_75k_dataset()
        df_base.to_csv(dataset_file, index=False)

    logger.info("Loading merged ground-truth dataset (CropHarvest + AgriFieldNet + Synthetic)...")
    merged_df = load_merged_ground_truth(
        use_cropharvest=True,
        use_agrifieldnet=True,
        use_synthetic=True,
        cropharvest_max_samples=5000,
    )

    train_and_save_model(merged_df)
