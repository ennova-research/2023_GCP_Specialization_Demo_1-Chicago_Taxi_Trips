import aiplatform

def endpoint_predict_sample(
    project: str, location: str, instances: list, endpoint: str
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    print(prediction)
    return prediction

result = endpoint_predict_sample(
    project="ml-spec",
    location="europe-west4",
    instances=[{
      "model_timestamp": "2023-12-07",
      "day": "2024-03-06"
    }],
    endpoint="1858495708336750592",
)

print(result)