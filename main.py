import pydantic
import uvicorn

from demo_lib import create_masked_model
from demo_lib import forecast
from demo_lib import load_model
from demo_lib import retrain_model
from demo_lib import save_model

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks

from typing import List
from typing import Union
import os

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"

app = FastAPI()

class NewModelBody(pydantic.BaseModel):
    y:Union[List, None]=None
    covid_start:int=2628
    covid_stop:int=3500
    num_variational_steps_per_iter:int=1000
    learning_rates:List[float]=[1e-2, 1e-3, 1e-4]


class ForecastBody(pydantic.BaseModel):
    model_timestamp:Union[str, None]=None
    day:str

class PredictBody(pydantic.BaseModel):
    instances: List[ForecastBody]


@app.get("/")
def read_root():
    return "The app is online."


@app.post("/newmodel")
async def create_new_model(body:NewModelBody, background_tasks: BackgroundTasks):
    """
    Function to create a new model.

    Arguments:
    - y: list of observed values, or None. If None, the model will be trained on the saved data.
    - covid_start: the index of the first day of the COVID-19 period.
    - covid_stop: the index of the last day of the COVID-19 period.
    - num_variational_steps_per_iter: the number of epochs to run for each learning rate value.
    - learning_rates: the list of learning rate values to use.

    Returns:
    A service message that tells the user that the training is ongoing.
    """
    async def create_and_save_model():
        model, y_train_masked = create_masked_model(
            y=body.y,
            covid_start=body.covid_start,
            covid_stop=body.covid_stop,
            num_variational_steps_per_iter=body.num_variational_steps_per_iter,
            learning_rates=body.learning_rates
        )
        save_model(model, y_train_masked)
        print("Model saved.")

    background_tasks.add_task(create_and_save_model)

    print("Model saved.")
    return {"message": "Training has started and will be available soon"}


@app.post("/retrain")
async def async_retrain_model(background_tasks: BackgroundTasks):
    """
    Function to retrain the model new model.

    Returns:
    A service message that tells the user that the training is ongoing.
    """
    async def create_and_save_model():
        retrain_model()

    background_tasks.add_task(create_and_save_model)

    print("Model saved.")
    return {"message": "Reraining has started and will be available soon"}


@app.post("/predict")
async def predict(request: Request):
    """
    Function to forecast the number of taxi rides at a specific day.

    Arguments:
    - model_timestamp: the timestamp of the model to use in the format "YYYY-MM-DD", or None. If None, the latest model will be used.
    - day: the day to forecast the number of taxi rides for.

    Returns:
    number_of_rides: int, the number of taxi rides predicted for the desired day.
    """
    req_data = await request.json()
    body = req_data["instances"][0]

    if "model_timestamp" not in body:
        body["model_timestamp"] = None

    try:
        return {"predictions":  [forecast(
            *load_model(
                timestamp=body["model_timestamp"]
            ),
            body["day"]
            )]
        }
    except:
        raise HTTPException(status_code=500, detail="malformed request") 

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "alive"}


if __name__ == '__main__':
    uvicorn.run(app, port=5000, host="0.0.0.0")
