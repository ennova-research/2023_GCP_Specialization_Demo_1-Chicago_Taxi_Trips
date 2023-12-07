# 2023 - GCP ML Specialization
## Demo 1: Chicago Taxi Trips

This demo demonstrates how to use TensorFlow Probability to predict the amount of daily taxi rides in Chicago. This will be helpful for urban planning (e.g., powering up the public transportation).

The project contains a directory `notebook` containing the following scripts:
1. `eda.ipynb`, containing the Exploratory Data Analysis performed. There is also a `.py` version of this file.
2. `model_creation.ipynb`, which contains the procedure followed for creating our time-series model.

The `demo_lib` directory contains the library with all the functions used by this demo.

Finally, the `main.py` contains the FastAPI app used to deliver the services.

As a demo, we stayed minimal, but we expect to be able to significantly scale-up the predictive capabilities of our model in potential follow-ups of this project. This could be done, e.g., by:
1. Adding an additional seasonality component to our model, to increase the granularity of the predictions from daily to hourly.
2. Implementing different models for different districts of the city.
3. Extending our analysis to the average taxi speed in order to determine the amount of traffic...

## Installation

These instructions assume that you have both Docker and Google Clour SDK on your machine, and that you have access to the Google Cloud Platform `ml-spec` project. If you don't, make sure to have met these conditions before proceeding.

To deploy a new version of the app, first create a container by running:

```docker build -t demo-1-app .```

Then upload the container on Google Cloud Build by running:

```gcloud builds submit --tag gcr.io/ml-spec/demo-1-app```

Finally deploy the container on Google Cloud Run by running:

```gcloud run deploy --image gcr.io/ml-spec/demo-1-app --platform managed --port 5000 --memory 4G --cpu 2 --timeout 60m```

The app is currently deployed at the following link:

https://demo-1-b6lmpdo3cq-uc.a.run.app/docs