import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
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
        # Create a "Client" object
        client = bigquery.Client()

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
    is_missing = [False] * covid_start + [True] * (covid_stop - covid_start) + [False] * (y.size - covid_stop)
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
        models = sorted(set([x.split('/')[0] for x in models]))
        
        # Keep only model directories
        models = [x for x in models if (len(x) == 16) and (x[:6] == 'model_')]
    
        # Select the most recent
        model = models[-1]
    else:
        model = f'model_{timestamp}'
    
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
    - `model`: tfp.sts.Sum, the model to use for the forecast
    - `y`: np.ndarray, or tfp.sts.MaskedTimeSeries, the time series associated to the model
    - `day`: str, the day for forecasting the number of taxi rides
    
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




def save_model_to_checkpoint(model,
                             output_dir: str = './model_checkpoint',
                             max_to_keep: int = 1):
    """
    Saves the model's parameters to a checkpoint directory, specified by `output_dir`.

    This function creates a checkpoint for the given model and manages the saving of the model's parameters. 
    It leverages TensorFlow's `tf.train.Checkpoint` and `tf.train.CheckpointManager` to handle the checkpointing. 
    The function allows specifying the directory where the checkpoint will be saved (`output_dir`) and prints the save path. 
    It also provides control over the maximum number of checkpoints to keep via the `max_to_keep` parameter.

    Inputs:
    - model: A TensorFlow model whose parameters are to be saved. The model should have a `parameters` attribute that
      can be iterated over to access its parameters.
    - output_dir (str, optional): The directory where the model checkpoint will be saved. Defaults to './model_checkpoint'.
    - max_to_keep (int, optional): The maximum number of checkpoints to retain. Defaults to 1.

    Returns:
    - str: The path where the model checkpoint is saved.
    
    Example:
    ```python
    # Assuming `model` is a TensorFlow model instance, to save into a Cloud Storage Bucket
    save_path = save_model_to_checkpoint(
        model,
        output_dir='gs://my_bucket_name/my_model_checkpoints',
        max_to_keep=5)
    ```
    """

    # Create a dictionary to hold the model's parameters
    params = {param.name: param for param in model.parameters}

    # Create a checkpoint that will manage the saving of your model parameters
    ckpt = tf.train.Checkpoint(**params)

    # Save the checkpoint to a directory
    ckpt_manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=max_to_keep)

    # Save the parameters
    save_path = ckpt_manager.save()
    print("Model saved to: ", save_path)

    return save_path

def load_model_from_checkpoint(y_train: tfp.sts.MaskedTimeSeries,
                               ckpt_path: str = './model_checkpoint',
                               trend: bool = False):
    """
    Loads a model from a specified checkpoint path.

    This function first builds a model using the provided training data (`y_train`).
    It then restores the model's parameters from a specified checkpoint path (`ckpt_path`).
    The function supports loading from both a specific checkpoint file or a directory managed
    by `tf.train.CheckpointManager`, depending on how the checkpoint was saved.

    Inputs:
    - y_train (tfp.sts.MaskedTimeSeries): The training data used to build the model, typically time series data with optional masking.
    - ckpt_path (str, optional): The path to the checkpoint file or directory from which the model's parameters will be restored.
      Defaults to './model_checkpoint'.
    - trend (bool, optional): If True, it uses a trend component, if False a local level (which is the default).

    Returns:
    - A TensorFlow Probability model with parameters restored from the specified checkpoint.

    Example:
    ```python
    # Assuming `y_train` is a MaskedTimeSeries instance and checkpoints are stored
    model = load_model_from_checkpoint(y_train, ckpt_path='gs://my_bucket_name/my_model_checkpoints', trend=False)
    ```
    """

    model = build_model(y_train, trend=trend)

    # Recreate the checkpoint with the model's parameters
    params = {param.name: param for param in model.parameters}
    ckpt = tf.train.Checkpoint(**params)

    # Use the CheckpointManager if managing multiple checkpoints or directly restore
    ckpt.restore(ckpt_path).assert_consumed()

    return model


def get_last_day():
    """Get the last day with 1000+ rides from BigQuery view."""
    client = bigquery.Client()

    query_ride_counts = """
    SELECT
        *
    FROM
        `ml-spec.demo1.group_ride_counts_by_ymd`
    ORDER BY
        Year DESC,
        Month DESC,
        Day DESC
    LIMIT
        100
    """

    # Create the dataframe
    df = client.query(query_ride_counts).to_dataframe()
    mask = df['num_trips'] >= 1e3
    df = df.loc[mask]

    year = str(df.iloc[0]['Year'])
    month = str(df.iloc[0]['Month'])
    day = str(df.iloc[0]['Day'])

    if len(month) == 1:
        month = '0' + month
    if len(day) == 1:
        day = '0' + day

    return f"{year}-{month}-{day}"


def retrain_model():

    # Load the latest model
    model, y_train = load_model()

    # Dump it to checkpoint for creating the saved_model
    checkpoint_dir = os.path.join('tmp', 'model_checkpoint')
    checkpoint_path = save_model_to_checkpoint(model, checkpoint_dir)

    del model

    # Create the wrapped model to save
    wrapper_model = TFPModelWrapper(build_model,
                                    y_train,
                                    last_train_day=get_last_day(),
                                    trend=False,
                                    variational_posteriors_samples=10000)

    # Update the model parameters with those from the train
    wrapper_model.load_parameters(checkpoint_path)

    # Save model to ../bin/saved_model
    export_dir = os.path.join('tmp', 'saved_model')
    wrapper_model.save(export_dir)

    storage_client = storage.Client()
    bucket = storage_client.bucket('engo-ml_spec2023-demo1')

    for local_root, _, files in os.walk(export_dir):
        for filename in files:
            local_path = os.path.join(local_root, filename)
            gcs_path = os.path.join('saved_model_new', filename).replace('\\', '/')

            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
    
    shutil.rmtree('tmp')


class TFPModelWrapper(tf.Module):
    """
    A wrapper class for TensorFlow Probability (TFP) models, needed to save them in `tf.saved_model` format.
    
    Attributes:
        model_fn: A function that returns a TFP model when called.
        args: Positional arguments to pass to `model_fn`.
        kwargs: Keyword arguments to pass to `model_fn`.
        model: The TFP model created by calling `model_fn`.
        q_samples: Samples from the variational posterior of the model.
        y: The observed time series data used by the model.
    """

    def __init__(self,
                 model_fn,
                 y_train: tfp.sts.MaskedTimeSeries,
                 trend:bool=False,
                 last_train_day:str="2023-10-31",
                 variational_posteriors_samples:int=1000):
        """
        Initializes the TFPModelWrapper instance with the specified model function, training data, and configuration options.

        The `model_fn` is expected to return a TensorFlow Probability Structural Time Series model configured according to the provided training data and trend setting. The instance then builds a variational posterior for the model, samples from it, and stores these samples for future forecasting.

        Args:
            model_fn: A callable that, when invoked with the training data and trend option, returns a tfp.sts.Sum model comprising of all the time series components.
            y_train: A tfp.sts.MaskedTimeSeries instance representing the observed time series data to be used for training the model.
            trend: A boolean flag indicating whether to include a trend component in the time series model. If `False` (default value), instead of a Trend component a Local Level is used.
            last_train_day: A string representing the last day of training data in "YYYY-MM-DD" format. Default value is "2023-10-31".
            variational_posteriors_samples: An int representing the number of samples to use for sampling the variational posteriors. Default value is 1000.
        """
        super(TFPModelWrapper, self).__init__()
        self.model_fn = model_fn
        self.last_train_day = last_train_day
        self.model = self.model_fn(y_train, trend=trend)
        variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=self.model)
        self.q_samples = variational_posteriors.sample(variational_posteriors_samples)
        self.y = y_train

    def load_parameters(self, checkpoint_path):
        """
        Loads model parameters from a TensorFlow checkpoint.

        Args:
            checkpoint_path: A string specifying the path to the TensorFlow checkpoint.
        """
        params = {param.name: param for param in self.model.parameters}
        ckpt = tf.train.Checkpoint(**params)
        ckpt.restore(checkpoint_path).assert_consumed()

    def forecast(self, num_steps_forecast=1):
        """Forecast future values of the time series data."""

        forecast_dist = tfp.sts.forecast(
            self.model,
            observed_time_series=self.y,
            parameter_samples=self.q_samples,
            num_steps_forecast=num_steps_forecast)
        forecast_mean = forecast_dist.mean()[..., tf.newaxis]

        # Convert back to linear values
        res = tf.round(10**forecast_mean[-1]) - 1

        return res

    def is_leap_year(self, year):
        """Determine if a year is a leap year."""
        return tf.logical_or(
            tf.equal(year % 400, 0),
            tf.logical_and(tf.equal(year % 4, 0), tf.not_equal(year % 100, 0))
        )

    def days_in_month(self, month, year):
        """Return the number of days in a month accounting for leap years."""
        return tf.switch_case(month, branch_fns={
            0: lambda: tf.constant(31),  # January
            1: lambda: tf.where(self.is_leap_year(year), tf.constant(29), tf.constant(28)),  # February
            2: lambda: tf.constant(31),  # March
            3: lambda: tf.constant(30),  # April
            4: lambda: tf.constant(31),  # May
            5: lambda: tf.constant(30),  # June
            6: lambda: tf.constant(31),  # July
            7: lambda: tf.constant(31),  # August
            8: lambda: tf.constant(30),  # September
            9: lambda: tf.constant(31),  # October
            10: lambda: tf.constant(30),  # November
            11: lambda: tf.constant(31),  # December
        }, default=lambda: tf.constant(0))

    def date_to_days(self, year, month, day):
        """Convert a date to a cumulative number of days accounting for leap years and varying month lengths."""
        days = tf.constant(0, dtype=tf.int32)
        for m in tf.range(1, 13):  # For each month
            days += tf.where(m < month, self.days_in_month(m - 1, year), 0)
        return year * 365 + (year // 4 - year // 100 + year // 400) + days + day

    def days_difference_tensor(self, target_date_str, given_date_str):
        """Compute the difference in days between two dates in format "YYYY-MM-DD"."""
        # Split the date strings into components
        target_split = tf.strings.split(target_date_str, '-')
        given_split = tf.strings.split(given_date_str, '-')

        # Convert string components to integers
        target_year = tf.strings.to_number(target_split[0], out_type=tf.int32)
        target_month = tf.strings.to_number(target_split[1], out_type=tf.int32)
        target_day = tf.strings.to_number(target_split[2], out_type=tf.int32)
        given_year = tf.strings.to_number(given_split[0], out_type=tf.int32)
        given_month = tf.strings.to_number(given_split[1], out_type=tf.int32)
        given_day = tf.strings.to_number(given_split[2], out_type=tf.int32)

        # Convert dates to a cumulative number of days
        target_days = self.date_to_days(target_year, target_month, target_day)
        given_days = self.date_to_days(given_year, given_month, given_day)

        # Calculate the difference in days
        difference = target_days - given_days

        return difference
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 1), dtype=tf.string)])
    def serving_default(self, instances):
        """
        Serves as the default method for making predictions. It takes a tensor containing a date string,
        calculates the days difference from a given date, and makes a forecast based on the difference.

        Args:
            instances: A tf.Tensor of shape (1, 1) and dtype tf.string, containing a date string in the format "YYYY-MM-DD".

        Returns:
            A tf.Tensor containing the forecasted value(s) based on the input date.
        """

        pred = self.forecast(self.days_difference_tensor(instances[0][0], self.last_train_day))

        return pred
    
    def save(self, export_dir):
        """Save the model to the given path."""
        tf.saved_model.save(self, export_dir)
        print(f'Model saved in {export_dir}.')

