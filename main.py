import pydantic
import uvicorn

from demo_lib import create_masked_model
from demo_lib import forecast
from demo_lib import load_model
from demo_lib import save_model

from fastapi import FastAPI

from typing import List
from typing import Union


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


@app.get("/")
def read_root():
    return "The app is online."


@app.post("/newmodel")
def create_new_model(body:NewModelBody):
    """
    Function to create a new model.

    Arguments:
    - y: list of observed values, or None. If None, the model will be trained on the saved data.
    - covid_start: the index of the first day of the COVID-19 period.
    - covid_stop: the index of the last day of the COVID-19 period.
    - num_variational_steps_per_iter: the number of epochs to run for each learning rate value.
    - learning_rates: the list of learning rate values to use.

    Returns:
    200 if a new model is created. The model created will be saved as "model_YYYY-MM-DD".
    """
    model, y_train_masked = create_masked_model(
        y=body.y,
        covid_start=body.covid_start,
        covid_stop=body.covid_stop,
        num_variational_steps_per_iter=body.num_variational_steps_per_iter,
        learning_rates=body.learning_rates
    )
    save_model(model, y_train_masked)
    return 200


@app.post("/forecast")
def forecast_future_values(body:ForecastBody):
    """
    Function to forecast the number of taxi rides at a specific day.

    Arguments:
    - model_timestamp: the timestamp of the model to use in the format "YYYY-MM-DD", or None. If None, the latest model will be used.
    - day: the day to forecast the number of taxi rides for.

    Returns:
    number_of_rides: int, the number of taxi rides predicted for the desired day.
    """
    return forecast(
        *load_model(
            timestamp=body.model_timestamp
        ),
        body.day
    )


if __name__ == '__main__':
    uvicorn.run(app, port=5000, host="0.0.0.0")
