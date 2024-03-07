gcloud builds submit \
  --config cloudbuild.yaml . \
  --project ml-spec \
  --timeout=60m