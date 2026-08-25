"""
AgriN — Train Demo Crop Classifier

Trains a Random Forest classifier on synthetic demo data.
This produces a working model for development and demonstration.

Usage:
    python scripts/train_demo_model.py

NOTE: For real crop classification, you need ground-truth labeled data
from the pilot region. This script uses synthetic data only.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.crop_classifier import CropClassifierService, generate_demo_training_data


def main():
    print("=" * 60)
    print("AgriN — Training Demo Crop Classifier")
    print("=" * 60)
    print()
    print("⚠️  This uses SYNTHETIC training data for demonstration.")
    print("    For real classification, provide ground-truth labels.")
    print()

    # Generate demo data
    print("Generating synthetic training data (300 samples)...")
    features, labels = generate_demo_training_data(n_samples=300)
    print(f"  Wheat: {labels.count('wheat')}")
    print(f"  Rice:  {labels.count('rice')}")
    print(f"  Other: {labels.count('other')}")
    print()

    # Train
    print("Training Random Forest classifier...")
    classifier = CropClassifierService()
    metrics = classifier.train(features, labels)

    print()
    print("📊 Evaluation Metrics:")
    print(f"  Accuracy:         {metrics['accuracy']:.3f}")
    print(f"  Precision (avg):  {metrics['precision_weighted']:.3f}")
    print(f"  Recall (avg):     {metrics['recall_weighted']:.3f}")
    print(f"  F1-Score (avg):   {metrics['f1_weighted']:.3f}")
    print()
    print("Classification Report:")
    print(metrics["classification_report"])
    print()
    print("Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")
    print()

    # Feature importance
    importance = classifier.get_feature_importance()
    print("Feature Importance (top 5):")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, val in sorted_imp:
        print(f"  {name}: {val:.4f}")
    print()

    print("✅ Model saved to models/crop_classifier/")
    print("   The dashboard will automatically load this model.")


if __name__ == "__main__":
    main()
