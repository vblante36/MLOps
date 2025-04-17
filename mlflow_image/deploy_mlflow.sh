#!/bin/bash

# ========= CONFIGURATION ========= #
PROJECT_ID="mlops-2025-456701"
REGION="us-west2"
REPO_NAME="mlflow-docker"
SERVICE_NAME="mlflow-server"
TAG="v1"
ACCOUNT_NAME="mlflow-service-acct"
IMAGE="us-west2-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/mlflow:$TAG"

# ========= STEP 1: Build Docker Image ========= #
echo "> Building Docker image..."
docker build --platform linux/amd64 -t "$IMAGE" . || {
  echo "❌ Docker build failed"
  exit 1
}

# ========= STEP 2: Push Image to Artifact Registry ========= #
echo "> Pushing Docker image to Artifact Registry..."
docker push "$IMAGE" || {
  echo "❌ Docker push failed"
  exit 1
}

# ========= STEP 3: Deploy to Cloud Run ========= #
echo "> Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "$ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
  --update-secrets=/secrets/credentials=access_keys:latest \
  --update-secrets=POSTGRESQL_URL=database_url:latest \
  --update-secrets=STORAGE_URL=bucket_url:latest \
  --memory 2Gi \
  --allow-unauthenticated \
  --port 8080 || {
  echo "❌ Deployment failed"
  exit 1
}

# ========= DONE ========= #
echo "\n✅ MLflow deployed successfully to Cloud Run!"
