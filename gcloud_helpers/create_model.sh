gcloud beta ai models upload \
  --region=europe-west4 \
  --display-name=demo-1-model \
  --container-image-uri=gcr.io/ml-spec/demo-1-app:latest \
  --container-ports=5000 \
  --container-health-route=/healthcheck \
  --container-predict-route=/predict