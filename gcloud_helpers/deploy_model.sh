export VERTEX_AI_ENDPOINT_ID="$(gcloud ai endpoints list --region=europe-west4 | grep demo-1 | awk '{print $1}')"
export VERTEX_AI_MODEL_ID="$(gcloud ai models list --region=europe-west4 | grep demo-1 | awk '{print $1}')"

gcloud ai endpoints deploy-model  ${VERTEX_AI_ENDPOINT_ID} \
  --region=europe-west4 \
  --model=${VERTEX_AI_MODEL_ID} \
  --display-name=demo-1-app\
  --machine-type=n1-standard-2 \
  --accelerator=count=1,type=nvidia-tesla-t4 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --traffic-split=0=100 \
  --enable-access-logging \
  --enable-container-logging