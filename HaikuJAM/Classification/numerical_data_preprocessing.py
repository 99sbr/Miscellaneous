import numpy as np
import pandas as pd
from fancyimpute import IterativeImputer as MICE

mice = MICE(n_nearest_features=30)


def preprocessing_numerics_df(numerical_df):
    numerical_df.Host_Since.ffill(axis=0, inplace=True)
    numerical_df['Host_Since'] = pd.to_datetime(numerical_df['Host_Since'],
                                                format='%Y-%m-%d',
                                                errors='coerce')

    numerical_df.Calendar_last_Scraped.ffill(axis=0, inplace=True)
    numerical_df['Calendar_last_Scraped'] = pd.to_datetime(
        numerical_df['Calendar_last_Scraped'], format='%Y-%m-%d', errors='coerce')

    numerical_df.First_Review.ffill(axis=0, inplace=True)
    numerical_df['First_Review'] = pd.to_datetime(numerical_df['First_Review'],
                                                  format='%Y-%m-%d',
                                                  errors='coerce')

    numerical_df.Last_Review.ffill(axis=0, inplace=True)
    numerical_df['Last_Review'] = pd.to_datetime(numerical_df['Last_Review'],
                                                 format='%Y-%m-%d',
                                                 errors='coerce')

    numerical_df['Host_Since_dayofyear'] = numerical_df.Host_Since.dt.dayofyear
    numerical_df['Host_Since_weekday'] = numerical_df.Host_Since.dt.weekday
    numerical_df['Host_Since_week'] = numerical_df.Host_Since.dt.week
    numerical_df['Host_Since_quarter'] = numerical_df.Host_Since.dt.quarter

    numerical_df[
        'Calendar_last_Scraped_dayofyear'] = numerical_df.Calendar_last_Scraped.dt.dayofyear
    numerical_df[
        'Calendar_last_Scraped_weekday'] = numerical_df.Calendar_last_Scraped.dt.weekday
    numerical_df[
        'Calendar_last_Scraped_week'] = numerical_df.Calendar_last_Scraped.dt.week
    numerical_df[
        'Calendar_last_Scraped_quarter'] = numerical_df.Calendar_last_Scraped.dt.quarter

    numerical_df['First_Review_dayofyear'] = numerical_df.First_Review.dt.dayofyear
    numerical_df[
        'First_Review_Scraped_weekday'] = numerical_df.First_Review.dt.weekday
    numerical_df['First_Review_Scraped_week'] = numerical_df.First_Review.dt.week
    numerical_df['First_Review_quarter'] = numerical_df.First_Review.dt.quarter

    numerical_df['Last_Review_dayofyear'] = numerical_df.Last_Review.dt.dayofyear
    numerical_df['Last_Review_weekday'] = numerical_df.Last_Review.dt.weekday
    numerical_df['Last_Review_week'] = numerical_df.Last_Review.dt.week
    numerical_df['Last_Review_quarter'] = numerical_df.Last_Review.dt.quarter

    numerical_df.drop([
        'Host_Since', 'Calendar_last_Scraped', 'First_Review', 'Last_Review',
        'Geolocation'
    ],
        1,
        inplace=True)

    for column in numerical_df.columns:
        numerical_df[column] = pd.to_numeric(numerical_df[column], errors='coerce')

    numerical_df['Rating_log'] = np.log(numerical_df.Rating)

    numerical_df.drop('Rating', 1, inplace=True)

    numerical_df['coord_x'] = np.cos(numerical_df['Latitude']) * np.cos(
        numerical_df['Longitude'])
    numerical_df['coord_y'] = np.cos(numerical_df['Latitude']) * np.sin(
        numerical_df['Longitude'])
    numerical_df['coord_z'] = np.sin(numerical_df['Latitude'])

    numerical_df.drop(['Latitude', 'Longitude'], 1, inplace=True)

    all_numerical_df = pd.DataFrame(data=mice.fit_transform(numerical_df),
                                    columns=numerical_df.columns)

    return all_numerical_df
