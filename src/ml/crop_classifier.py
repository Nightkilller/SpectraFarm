"""
AgriN — Crop Classifier

Random Forest crop classification: Wheat / Rice / Other.

Supports:
- Training on extracted features with labels
- Saving/loading trained models
- Prediction with confidence scores
- Feature importance extraction
- Model evaluation metrics
- Demo mode with a pre-generated synthetic model
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.config.settings import MODELS_DIR, get_settings
from src.data.schemas import CropPrediction, CropType, DataSource

logger = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "crop_classifier"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"

# Default class mapping
CLASS_MAP = {
    0: CropType.GRAM,
    1: CropType.LENTIL,
    2: CropType.MAIZE,
    3: CropType.MUSTARD,
    4: CropType.POTATO,
    5: CropType.RICE,
    6: CropType.SUGARCANE,
    7: CropType.WHEAT,
}
LABEL_MAP = {
    "wheat": 7, "rice": 5, "mustard": 3, "sugarcane": 6,
    "potato": 4, "lentil": 1, "maize": 2, "gram": 0, "other": 7
}


class CropClassifierService:
    """Random Forest crop classifier with train/predict/evaluate capabilities."""

    def __init__(self) -> None:
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: list[str] = []
        self.label_encoder = None
        self.class_map = CLASS_MAP.copy()
        self.model_version = "rf_v1.0_colab"
        self._load_model()

    def _load_model(self) -> None:
        """Try to load a previously trained model."""
        if MODEL_PATH.exists() and FEATURE_NAMES_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                self.feature_names = joblib.load(FEATURE_NAMES_PATH)

                if LABEL_ENCODER_PATH.exists():
                    self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                    # Dynamically map label encoder classes to CropType
                    for idx, cname in enumerate(self.label_encoder.classes_):
                        clean_name = str(cname).lower()
                        try:
                            self.class_map[idx] = CropType(clean_name)
                        except ValueError:
                            self.class_map[idx] = CropType.OTHER

                logger.info(f"Loaded crop classifier from {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.model = None

    def is_trained(self) -> bool:
        return self.model is not None

    def train(
        self,
        features: list[dict[str, float]],
        labels: list[str],
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
    ) -> dict:
        """
        Train the Random Forest classifier.

        Args:
            features: List of feature dicts (one per sample)
            labels: Crop labels ("wheat", "rice", "other")
            test_size: Fraction for validation split
            random_state: Random seed
            n_estimators: Number of trees

        Returns:
            Dict with evaluation metrics
        """
        if not features or not labels:
            raise ValueError("Features and labels must not be empty")

        # Get consistent feature names from first sample
        self.feature_names = sorted(features[0].keys())

        # Convert to numpy arrays
        X = np.array([[f.get(name, 0.0) for name in self.feature_names] for f in features])
        y = np.array([LABEL_MAP.get(label, 2) for label in labels])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            max_depth=15,
            min_samples_split=5,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = self._compute_metrics(y_test, y_pred)

        # Save model
        self._save_model()

        logger.info(f"Crop classifier trained. Accuracy: {metrics['accuracy']:.3f}")
        return metrics

    def predict(
        self,
        features: dict[str, float],
        farm_id: str,
    ) -> CropPrediction:
        """
        Predict crop type from features.

        Falls back to demo prediction if model is not trained.
        """
        if not self.is_trained():
            logger.warning("No trained model — returning demo prediction")
            from src.data.demo_data import get_demo_crop_prediction
            return get_demo_crop_prediction(farm_id)

        # Build feature vector in correct order
        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])

        # Predict with probabilities
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        predicted_crop = self.class_map.get(prediction, CropType.OTHER)
        confidence = float(probabilities[prediction])

        # Feature importance
        importances = dict(zip(
            self.feature_names,
            [round(float(v), 4) for v in self.model.feature_importances_],
        ))

        return CropPrediction(
            farm_id=farm_id,
            predicted_crop=predicted_crop,
            confidence=round(confidence, 3),
            model_version=self.model_version,
            prediction_date=date.today(),
            feature_importance=importances,
            data_source=DataSource.LIVE,
        )

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance from the trained model."""
        if not self.is_trained():
            return {}

        return dict(zip(
            self.feature_names,
            [round(float(v), 4) for v in self.model.feature_importances_],
        ))

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute classification evaluation metrics."""
        class_names = ["Wheat", "Rice", "Other"]
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, target_names=class_names, zero_division=0
            ),
        }

    def _save_model(self) -> None:
        """Save the trained model and feature names."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.feature_names, FEATURE_NAMES_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")


def generate_demo_training_data(n_samples: int = 300) -> tuple[list[dict], list[str]]:
    """
    Generate synthetic training data for the crop classifier.

    This is for DEMO/DEVELOPMENT purposes only.  Real training requires
    ground-truth labeled field data.
    """
    np.random.seed(42)
    features_list = []
    labels = []

    for _ in range(n_samples):
        crop = np.random.choice(["wheat", "rice", "other"], p=[0.4, 0.4, 0.2])

        if crop == "wheat":
            f = {
                "ndvi_mean": np.random.normal(0.55, 0.08),
                "ndvi_std": np.random.normal(0.12, 0.03),
                "ndvi_min": np.random.normal(0.25, 0.06),
                "ndvi_max": np.random.normal(0.78, 0.06),
                "ndvi_range": np.random.normal(0.53, 0.08),
                "ndvi_trend": np.random.normal(-0.02, 0.01),
                "ndwi_mean": np.random.normal(0.05, 0.04),
                "red_mean": np.random.normal(0.06, 0.015),
                "green_mean": np.random.normal(0.07, 0.015),
                "nir_mean": np.random.normal(0.28, 0.05),
                "swir1_mean": np.random.normal(0.18, 0.04),
                "vv_mean": np.random.normal(-11.0, 1.5),
                "vh_mean": np.random.normal(-17.5, 1.5),
            }
        elif crop == "rice":
            f = {
                "ndvi_mean": np.random.normal(0.62, 0.09),
                "ndvi_std": np.random.normal(0.15, 0.04),
                "ndvi_min": np.random.normal(0.15, 0.05),
                "ndvi_max": np.random.normal(0.82, 0.05),
                "ndvi_range": np.random.normal(0.67, 0.07),
                "ndvi_trend": np.random.normal(0.01, 0.015),
                "ndwi_mean": np.random.normal(0.15, 0.06),
                "red_mean": np.random.normal(0.05, 0.012),
                "green_mean": np.random.normal(0.06, 0.012),
                "nir_mean": np.random.normal(0.32, 0.06),
                "swir1_mean": np.random.normal(0.14, 0.04),
                "vv_mean": np.random.normal(-12.5, 1.5),
                "vh_mean": np.random.normal(-19.0, 1.5),
            }
        else:
            f = {
                "ndvi_mean": np.random.normal(0.35, 0.12),
                "ndvi_std": np.random.normal(0.08, 0.03),
                "ndvi_min": np.random.normal(0.18, 0.06),
                "ndvi_max": np.random.normal(0.52, 0.10),
                "ndvi_range": np.random.normal(0.34, 0.10),
                "ndvi_trend": np.random.normal(0.0, 0.02),
                "ndwi_mean": np.random.normal(-0.02, 0.06),
                "red_mean": np.random.normal(0.08, 0.02),
                "green_mean": np.random.normal(0.09, 0.02),
                "nir_mean": np.random.normal(0.20, 0.06),
                "swir1_mean": np.random.normal(0.22, 0.05),
                "vv_mean": np.random.normal(-10.0, 2.0),
                "vh_mean": np.random.normal(-16.5, 2.0),
            }

        features_list.append(f)
        labels.append(crop)

    return features_list, labels
