#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from shapely import wkt

from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler

dataset_id = "chicago_taxi_trips"
project = "bigquery-public-data"
table_id = "taxi_trips"

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the dataset
dataset_ref = client.dataset(dataset_id=dataset_id,
                             project=project)

# Construct a reference to the table
table_ref = dataset_ref.table(table_id=table_id)

# Fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "full" table
df = client.list_rows(table, max_results=5).to_dataframe()
df

#%%

# Get the number of rides per year
query_rides_per_year = """
SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 
       COUNT(1) AS num_trips
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
GROUP BY year 
ORDER BY year;
"""
query_job_rides_per_year = client.query(query_rides_per_year)

# API request - run the query, and return a pandas DataFrame
df_rides_per_year = query_job_rides_per_year.to_dataframe()
df_rides_per_year = df_rides_per_year.rename(columns={'num_trips': 'Number of Trips'}).set_index('year')

# Plot the results
fig, ax = plt.subplots(1, figsize=(8, 4.5))
sns.lineplot(df_rides_per_year, ax=ax)

#%%

query_rides_per_year_month = """
SELECT
  EXTRACT(YEAR FROM trip_start_timestamp) AS year,
  EXTRACT(MONTH FROM trip_start_timestamp) AS month,
  COUNT(1) AS num_trips
FROM
  `bigquery-public-data.chicago_taxi_trips.taxi_trips`
GROUP BY
  year, month
ORDER BY
  year, month;
"""

# Set up the query
query_job_rides_per_year_month = client.query(query_rides_per_year_month) 

# API request - run the query, and return a pandas DataFrame
df_rides_per_year_month = query_job_rides_per_year_month.to_dataframe() 
df_rides_per_year_month['timestamp'] = pd.to_datetime(df_rides_per_year_month[['year', 'month']].assign(DAY=1))
df_rides_per_year_month = df_rides_per_year_month.rename(columns={'num_trips': 'Number of Trips'})

# Plot the results
fig, ax = plt.subplots(1, figsize=(8, 4.5))
sns.lineplot(df_rides_per_year_month['Number of Trips'], ax=ax)

#%%

df_rides_per_year_month_heatmap = df_rides_per_year_month.pivot(index='year', columns='month', values='Number of Trips')
df_rides_per_year_month_heatmap = df_rides_per_year_month_heatmap.fillna(0.).astype(int)
df_rides_per_year_month_heatmap.index.name = 'Year'
df_rides_per_year_month_heatmap.columns.name = 'Month'

fig, ax = plt.subplots(1, figsize=(8, 4.5))
sns.heatmap(data=df_rides_per_year_month_heatmap, ax=ax)

#%%

fig, ax = plt.subplots(1, figsize=(8, 4.5))
for year in sorted(set(df_rides_per_year_month['year'])):
    mask = (df_rides_per_year_month['year'] == year)
    data = df_rides_per_year_month.loc[mask].set_index('month')['Number of Trips']
    sns.lineplot(data, label=year, ax=ax)

#%%

scaler = StandardScaler()

fig, ax = plt.subplots(1, figsize=(8, 4.5))
for year in sorted(set(df_rides_per_year_month['year'])):
    mask = (df_rides_per_year_month['year'] == year)
    data = df_rides_per_year_month.loc[mask].set_index('month')['Number of Trips']
    data = pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten(),
                     index=data.index,
                     name='Number of Trips')
    sns.lineplot(data, label=year, ax=ax)

#%%

query_rides_per_ymdh = """
SELECT
  EXTRACT(YEAR FROM trip_start_timestamp) AS Year,
  EXTRACT(MONTH FROM trip_start_timestamp) AS Month,
  EXTRACT(DAY FROM trip_start_timestamp) AS Day,
  EXTRACT(HOUR FROM trip_start_timestamp) AS Hour,
  COUNT(1) AS num_trips,
FROM
  `bigquery-public-data.chicago_taxi_trips.taxi_trips`
GROUP BY
  year, month, day, hour
ORDER BY
  year, month, day, hour;
"""

# Set up the query
query_job_rides_per_ymdh = client.query(query_rides_per_ymdh) 

# API request - run the query, and return a pandas DataFrame
df_rides_per_ymdh = query_job_rides_per_ymdh.to_dataframe() 
df_rides_per_ymdh['timestamp'] = pd.to_datetime(df_rides_per_ymdh[['Year', 'Month', 'Day', 'Hour']])
df_rides_per_ymdh = df_rides_per_ymdh.rename(columns={'num_trips': 'Number of Trips'})
df_rides_per_ymdh['Week Day'] = pd.to_datetime(df_rides_per_ymdh['timestamp']).dt.day_name()

# Plot the results
# fig, ax = plt.subplots(1, figsize=(8, 4.5))
# sns.lineplot(df_rides_per_ymdh['Number of Trips'], ax=ax)

df_rides_per_ymdh.groupby(['Week Day', 'Hour']).agg({'Number of Trips': sum}).reset_index().pivot(index='Hour', columns='Week Day', values='Number of Trips')

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

#%%
fig, ax = plt.subplots(1, figsize=(16, 9))
sns.lineplot(data=df_rides_per_ymdh,
             x='Hour', y='Number of Trips',
             hue='Week Day', hue_order=weekdays,
             errorbar=('se', 2),  # 2 standard deviations
             ax=ax)
ax.legend(title='Week Day', ncol=2)

#%%

query_ride_counts = """
SELECT
  EXTRACT(YEAR FROM trip_start_timestamp) AS Year,
  EXTRACT(MONTH FROM trip_start_timestamp) AS Month,
  EXTRACT(DAY FROM trip_start_timestamp) AS Day,
  EXTRACT(HOUR FROM trip_start_timestamp) AS Hour,
  COUNT(1) AS num_trips,
  pickup_community_area,
  dropoff_community_area
FROM
  `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE
  pickup_community_area IS NOT NULL
AND
  dropoff_community_area IS NOT NULL
GROUP BY
  Year, Month, Day, Hour, pickup_community_area, dropoff_community_area
ORDER BY
  Year, Month, Day, Hour;
"""

df = client.query(query_ride_counts).to_dataframe()
df.rename(columns={'num_trips': 'Number of Trips'}, inplace=True)
df['Week Day'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']]).dt.day_name()

#%%

mask = (df['Week Day'] == 'Monday')

fig, ax = plt.subplots(1, figsize=(8, 4.5))
sns.lineplot(data=df.loc[mask],
             x='Hour', y='Number of Trips',
             hue=df['dropoff_community_area'].astype(str),
             errorbar=('se', 2),  # 2 standard deviations
             ax=ax)
ax.legend(title='Week Day', ncol=2)

#%%

df_areas = pd.read_csv(os.path.join('..', 'data', 'raw', 'CommAreas.csv'), sep=';')
df_areas.columns = df_areas.columns.str.lower()
df_areas['Lat. Centroid'] = df_areas['the_geom'].apply(lambda x: wkt.loads(x).centroid.x)
df_areas['Long. Centroid'] = df_areas['the_geom'].apply(lambda x: wkt.loads(x).centroid.y)
df_areas = df_areas[['area_numbe', 'Lat. Centroid', 'Long. Centroid']]
df_areas = df_areas.rename(columns={'area_numbe': 'Comm. Area'})
df_areas = df_areas.set_index('Comm. Area').sort_index()
df_areas

lat = df_areas['Lat. Centroid'].to_dict()
long = df_areas['Long. Centroid'].to_dict()

#%%

feature = 'pickup_community_area'
hour = 16
fig, ax = plt.subplots(1, figsize=(8, 4.5))
sns.scatterplot(data=df_areas, x='Lat. Centroid', y='Long. Centroid',
                hue = df.loc[df['Hour'] == hour][[feature, 'Number of Trips']].groupby(feature).agg({'Number of Trips': np.mean})['Number of Trips'].astype(float),
                ax=ax)

#%%

df_hm = df.pivot_table(values='Number of Trips',
                        index='pickup_community_area',
                        columns='dropoff_community_area',
                        aggfunc='sum'
                        )
df_hm = df_hm.fillna(0.).astype(int)
df_hm = np.log10(df_hm + 1)

fig, ax = plt.subplots(1, figsize=(8, 4.5))
sns.heatmap(data=df_hm, ax=ax)

