#!/bin/bash
export VERTEX_AI_ENDPOINT_ID="$(gcloud ai endpoints list --region=europe-west4 | grep demo-1 | awk '{print $1}')"
export PROJECT_ID="$(gcloud config get-value project)"
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://europe-west4-aiplatform.googleapis.com/v1/projects/ennova-ai-test/locations/europe-west4/endpoints/${VERTEX_AI_ENDPOINT_ID}:predict \
  -d test.json