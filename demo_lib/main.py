import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from typing import List
from typing import Union

from google.cloud import bigquery
from google.cloud import storage
from io import BytesIO


def build_model(y, trend=True):
    """
    Function that, given a time series `y` in the form of a numpy array, returns a model made by the following components:
    - a yearly seasonality;
    - a weekly seasonality;
    - a local linear trend (if `trend` is True), or
    - a local level (if `trend` is False).
    """
    # Number of days per month, starting from year 2013
    num_days_per_month = np.array(
        [[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
        [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
        [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
        [31, 29, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31]])

    # Yearly seasonality
    month_of_year = tfp.sts.Seasonal(
        num_seasons=12,
        num_steps_per_season=num_days_per_month,
        observed_time_series=y,
        name='month_of_year')

    # Weekly seasonality
    day_of_week = tfp.sts.Seasonal(
        num_seasons=7,
        num_steps_per_season=1,
        observed_time_series=y,
        name='day_of_week')

    if trend:
        # Local linear trend
        trend = tfp.sts.LocalLinearTrend(
            observed_time_series=y, name='trend')
        components = [trend, day_of_week, month_of_year]
    else:
        # Local level
        level = tfp.sts.LocalLevel(
            observed_time_series=y, name='level')
        components = [level, day_of_week, month_of_year]

    # Sum of the components
    model = tfp.sts.Sum(
        components=components,
        observed_time_series=y)
    
    return model


def train_tfp_model(y, model, num_variational_steps_per_iter=1000, learning_rates=[1e-2, 1e-3, 1e-4]):
    """
    Function to use variational inference for training a TensorFlow Probability model.
    It takes the following arguments:
    - `y`, the time series;
    - `model`, the TensorFlow Probability model;
    - `num_variational_steps_per_iter`, the number of epochs for each learning rate value (default is 1000);
    - `learning_rates`, a list with the sequence of the learning rates to use for the training (default is [1e-2, 1e-3, 1e-4]).
    """
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)
    for learning_rate in learning_rates:
        print(f'Learning rate: {learning_rate}')
        # Build and optimize the variational loss function.
        elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=model.joint_distribution(
                observed_time_series=y).log_prob,
            surrogate_posterior=variational_posteriors,
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            num_steps=num_variational_steps_per_iter,
            jit_compile=True)

        plt.plot(elbo_loss_curve)
        plt.show()

    return model


def create_masked_model(y:Union[List, None]=None,
                        covid_start:int=2628,
                        covid_stop:int=3500,
                        num_variational_steps_per_iter:int=1000,
                        learning_rates:List[float]=[1e-2, 1e-3, 1e-4]):

    if y is None:
        # Define the dataset id for importing the data
        dataset_id = "chicago_taxi_trips"

        # Create a "Client" object
        client = bigquery.Client()

        # Construct a reference to the dataset
        dataset_ref = client.dataset(dataset_id=dataset_id)

        query_ride_counts = """
        SELECT
          *
        FROM
          `ml-spec.demo1.group_ride_counts_by_ymdh`
        """

        # Create the temporary dataframe
        df_tmp = client.query(query_ride_counts).to_dataframe()
        df_tmp.rename(columns={'num_trips': 'Number of Trips'}, inplace=True)
        df_tmp = df_tmp.set_index(pd.to_datetime(df_tmp[['Year', 'Month', 'Day', 'Hour']]))

        # Create the empty dataframe
        df = pd.DataFrame(index=pd.date_range(start=df_tmp.index[0], end=df_tmp.index[-1], freq='D'))

        # Aggregate the temporary data in terms of the day
        num = df_tmp.groupby(
            pd.to_datetime(df_tmp[['Year', 'Month', 'Day', 'Hour']]).dt.strftime('%Y-%m-%d')
        ).agg({'Number of Trips': sum})
        num.index = pd.to_datetime(num.index)

        # Save the data in the dataframe
        df['Number of Trips'] = num
        df['Number of Trips'] = df['Number of Trips'].fillna(0).astype(int)

        # Delete the last datapoint, as it does not include a full month
        df = df.iloc[:-1]

        y = np.log10(df['Number of Trips'].values.astype(np.float32) + 1)
    else:
        y = np.array(y).astype(np.float32)

    # Mask the covid period
    is_missing = [False]* covid_start + [True] * (covid_stop - covid_start) + [False] * (y.size - covid_stop)
    y_masked = tfp.sts.MaskedTimeSeries(y, is_missing)

    # Create model
    model = build_model(y_masked, trend=False)

    # Train model
    masked_model = train_tfp_model(y_masked,
                                   model,
                                   num_variational_steps_per_iter=num_variational_steps_per_iter,
                                   learning_rates=learning_rates)
    
    return masked_model, y_masked


def get_timestamp():
    from datetime import datetime
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d")


def save_model(model, y):
    model_name = f'model_{get_timestamp()}'
    
    variable_dict = {'y': y}
    for component in model.components:
        variable_dict[component.name[:-1]] = component.init_parameters
    upload_model_to_gcs(variable_dict, model_name)
    
    return 'ok'


def upload_model_to_gcs(variable_dict, model_name):
    # Define the bucket-related variables
    bucket_name = 'engo-ml_spec2023-demo1'
    model_path = f'models/{model_name}'

    # Initialize a client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)
    
    for k, v in variable_dict.items():

        # Serialize the variable
        tmp_file = f'{k}.joblib'
        variable_content = joblib.dump(v, tmp_file)

        # Build the GCS object name by combining the base path and the variable name
        gcs_object_name = f"{model_path}/{k}.joblib"

        # Create a Blob object
        blob = bucket.blob(gcs_object_name)

        # Upload the local file to GCS
        blob.upload_from_filename(tmp_file)
        
        # Delete the local object
        os.remove(tmp_file)
    
    return 'ok'


def load_joblib_file_from_gcs(bucket_name, file_name):
    # Initialize a client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Get the blob (file) from the bucket
    blob = bucket.blob(file_name)

    # Download the file's content as bytes
    file_content_bytes = blob.download_as_bytes()

    # Load the joblib file from the bytes content
    content = BytesIO(file_content_bytes)
    loaded_data = joblib.load(content)

    return loaded_data


def path_exists_in_bucket(bucket_name, path):
    # Initialize a client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Check if the blob (file) with the specified path exists
    blob = bucket.blob(path)
    return blob.exists()


def list_files_in_bucket_path(bucket_name, prefix):
    # Initialize a client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # List all blobs (files) with the specified prefix (path) in the bucket
    blobs = bucket.list_blobs(prefix=prefix)

    # Extract and return the list of file names
    file_names = [blob.name for blob in blobs]
    return file_names



def load_model(timestamp=None):
    # Define the bucket-related variables
    bucket_name = 'engo-ml_spec2023-demo1'
    models_dir = 'models'

    if timestamp is None:
        # Get all the files at the model path
        models = list_files_in_bucket_path(bucket_name, models_dir)
        
        # Purge the names of the prefix
        models = [x[len(models_dir)+1:] for x in models]
        
        # Sort unique directory values
        models = sorted(set([x.split(os.sep)[0] for x in models]))
        
        # Keep only model directories
        models = [x for x in models if (len(x) == 16) and (x[:6] == 'model_')]
    
        # Select the most recent
        model = models[-1]
    
    # Define the model path
    model_path = f'{models_dir}/{model}'

    # Check if "trend" component exists
    trend = path_exists_in_bucket(bucket_name, f'{model_path}/trend.joblib')

    # Import the time series
    y = load_joblib_file_from_gcs(bucket_name, f'{model_path}/y.joblib')

    # Import the components
    day_of_week = tfp.sts.Seasonal(
        **load_joblib_file_from_gcs(bucket_name, f'{model_path}/day_of_week.joblib'))
    month_of_year = tfp.sts.Seasonal(
        **load_joblib_file_from_gcs(bucket_name, f'{model_path}/month_of_year.joblib'))
    if trend:
        trend = tfp.sts.LocalLinearTrend(
            **load_joblib_file_from_gcs(bucket_name, f'{model_path}/trend.joblib'))
        components = [trend, day_of_week, month_of_year]
    else:
        level = tfp.sts.LocalLevel(
            **load_joblib_file_from_gcs(bucket_name, f'{model_path}/level.joblib'))
        components = [level, day_of_week, month_of_year]

    # Sum of the components
    model = tfp.sts.Sum(
        components=components,
        observed_time_series=y)

    return model, y


def days_difference(target_date_str, given_date_str):
    # Convert strings to Pandas datetime objects
    target_date = pd.to_datetime(target_date_str)
    given_date = pd.to_datetime(given_date_str)

    # Calculate the difference in days
    difference = (target_date - given_date).days

    return difference


def forecast(model, y, day):
    """
    Function to forecast the taxi rides for a specific day.
    
    Inputs:
    `model`: tfp.sts.Sum, the model to use for the forecast
    `y`: np.ndarray, or tfp.sts.MaskedTimeSeries, the time series associated to the model
    `day`: str, the day for forecasting the number of taxi rides
    
    Returns an integer corresponding to the prediction.
    """
    
    assert isinstance(model, tfp.sts.Sum), "The model is not an instance of `tfp.sts.Sum`."
    assert isinstance(y, (tfp.sts.MaskedTimeSeries, np.ndarray)), "The model is not an instance of `tfp.sts.MaskedTimeSeries` or `np.ndarray`."

    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)
    q_samples = variational_posteriors.sample(100)

    # Number of data points to forecast for the final plot
    num_steps_forecast = days_difference(day, "2023-10-31")
    assert num_steps_forecast > 0, "The date must be posterior to 2023-10-31."

    # Forecast data
    forecast_dist = tfp.sts.forecast(
        model,
        observed_time_series=y,
        parameter_samples=q_samples,
        num_steps_forecast=num_steps_forecast)

    forecast_mean = forecast_dist.mean().numpy()[..., 0]

    return round(10**forecast_mean[-1]) - 1

