# AgriN — Cloud Deployment Guide

## Architecture

```
User Browser
     │
     ▼
Google Cloud Run (Streamlit Container)
     │
     ├──→ Google Earth Engine (Sentinel-2 / Sentinel-1 satellite data)
     ├──→ Trained ML Model (Random Forest .joblib in container)
     └──→ Google Gemini API (AI advisory generation)
```

---

## Prerequisites

1. **Google Cloud Project**: `agrin-506618` (or your own)
2. **APIs Enabled**:
   - Cloud Run API
   - Container Registry / Artifact Registry
   - Earth Engine API
   - Vertex AI / Generative Language API
3. **Service Account** with Earth Engine access (for headless auth in Cloud Run)
4. **gcloud CLI** installed and authenticated

---

## Step 1: Build the Docker Image

```bash
cd agriN

# Build locally
docker build -t agrin .

# Test locally
docker run -p 8080:8080 \
  -e GEMINI_API_KEY=your_key \
  -e GEE_PROJECT=agrin-506618 \
  agrin
```

Visit http://localhost:8080

---

## Step 2: Push to Google Artifact Registry

```bash
# Configure Docker for GCR
gcloud auth configure-docker asia-south1-docker.pkg.dev

# Tag
docker tag agrin asia-south1-docker.pkg.dev/agrin-506618/agrin/agrin:latest

# Push
docker push asia-south1-docker.pkg.dev/agrin-506618/agrin/agrin:latest
```

---

## Step 3: Deploy to Cloud Run

```bash
gcloud run deploy agrin \
  --image=asia-south1-docker.pkg.dev/agrin-506618/agrin/agrin:latest \
  --platform=managed \
  --region=asia-south1 \
  --port=8080 \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=3 \
  --allow-unauthenticated \
  --set-env-vars="GEMINI_API_KEY=your_key,GEE_PROJECT=agrin-506618,AGRIN_MODE=live" \
  --project=agrin-506618
```

---

## Step 4: Earth Engine Authentication for Cloud Run

For Cloud Run (headless), use a service account:

1. Create a service account with Earth Engine access:
   ```bash
   gcloud iam service-accounts create agrin-ee \
     --display-name="AgriN Earth Engine Service Account"
   ```

2. Register the service account with Earth Engine at:
   https://signup.earthengine.google.com/#!/service_accounts

3. Deploy Cloud Run with the service account:
   ```bash
   gcloud run deploy agrin \
     --service-account=agrin-ee@agrin-506618.iam.gserviceaccount.com \
     ...
   ```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GEE_PROJECT` | Google Cloud project ID for Earth Engine | Yes |
| `GEMINI_API_KEY` | Google Gemini API key | Optional (uses demo fallback) |
| `AGRIN_MODE` | `live` or `demo` | Optional (default: `live`) |

---

## Model Artifacts

The trained Random Forest model files must be present at:
```
models/crop_classifier/random_forest.joblib
models/crop_classifier/feature_names.joblib
```

These are included in the Docker image during build. To update:
1. Train new model in Google Colab (`notebooks/agrin_colab_training.py`)
2. Download model files
3. Place in `models/crop_classifier/`
4. Rebuild and redeploy the Docker image
