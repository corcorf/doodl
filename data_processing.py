import os
import pandas as pd
import numpy as np


def load_data(path='data'):
    """
    load the supermarket customer dataset
    return the loaded data as a dataframe
    """
    path = 'data'
    data = pd.DataFrame()
    for file in os.listdir(path):
        df = pd.read_csv(os.path.join(path, file), sep=';')
        df['timestamp'] = pd.to_datetime(df['timestamp'],
                                         format="%Y-%m-%d %H:%M:%S")
        data = pd.concat([data, df])
    data = data.sort_values(by=['timestamp']).reset_index(drop=True)
    return data


def get_aisles():
    """
    return a list of the aisles in the supermarket
    """
    aisles = ['fruit', 'spices', 'dairy', 'drinks', 'checkout']
    return aisles


def get_first_aisle_pmf(df_locations, day='all'):
    """
    get probabilities for first aisle visited
    """
    aisles = get_aisles()
    first_aisle_proba = pd.DataFrame(index=aisles)

    if day == 'all':
        first_aisle_proba['all days']\
            = df_locations.iloc[:, 0].value_counts()\
            / df_locations.iloc[:, 0].count()
    else:
        first_aisle_proba[f'day {day}']\
            = df_locations.loc[day].iloc[0].value_counts()\
            / df_locations.loc[day].iloc[0].count()

    first_aisle_proba = first_aisle_proba.fillna(0).squeeze()
    return first_aisle_proba


def get_customer_locations_by_time(data):
    """
    Return a dataframe containing the location of each customer at each time
    """
    df_location_by_time = data.sort_values(by=['customer_no', 'timestamp'])
    df_location_by_time['day'] = df_location_by_time['timestamp'].dt.dayofweek
    entrance_time = df_location_by_time.groupby(['day', 'customer_no'])\
                                       .nth(0)['timestamp']
    df_location_by_time = df_location_by_time.set_index(['day', 'customer_no'])
    time_elapsed = df_location_by_time['timestamp'] - entrance_time
    df_location_by_time['time_elapsed']\
        = time_elapsed.sort_index(level=1).values
    df_location_by_time\
        = df_location_by_time.reset_index()\
                             .set_index(
                                ['day', 'customer_no', 'time_elapsed']
                             )['location']
    time_range = pd.timedelta_range('00S', '3600S', freq='60S')
    df_location_by_time = df_location_by_time.unstack(-1)\
                                             .reindex(time_range, axis=1)
    df_location_by_time = df_location_by_time.fillna(method='ffill', axis=1)
    return df_location_by_time


def get_trans_matrix(df_location_by_time):
    """
    set up a dataframe for the transition matrix
    should have a column and a row for each aisle
    """
    aisles = get_aisles()
    transition_matrix_minutely = pd.DataFrame(columns=aisles,
                                              index=aisles[:-1])
    transition_matrix_minutely.index.rename('start aisle', inplace=True)
    transition_matrix_minutely.columns.rename('next aisle', inplace=True)

    # make a temporary version of the locations dataframe
    # this should have an extra column to avoid an indexing error later
    dummy_locations = df_location_by_time.copy()
    dummy_locations["23:59:59"] = np.nan

    # loop through the aisles, find the table indeces where that aisle appears
    # "next aisle" is the aisle named in the following column of the same row
    for a in aisles:
        instances_a = np.where(dummy_locations == a)
        instances_next = (instances_a[0], instances_a[1] + 1)
        transition_matrix_minutely.loc[a]\
            = pd.Series(dummy_locations.values[instances_next]).value_counts()

    # the total number of transitions from each aisle is the sum along the rows
    total_transitions = transition_matrix_minutely.sum(axis=1)

    # calculate the probability of each transition by dividing by the
    # total_transitions from each starting aisle
    transition_proba_minutely\
        = transition_matrix_minutely.div(total_transitions, axis=0).fillna(0)
    return transition_proba_minutely


def get_trans_matrix_by_day(data, day='all'):
    """
    calculate the transition matrix by day
    set up a dataframe for the transition matrix
    should have a column and a row for each aisle
    """
    aisles = get_aisles()
    day_idx = ['all'] + list(range(5))
    multi_idx = pd.MultiIndex.from_product([day_idx, aisles],
                                           names=['day', 'start aisle'])
    transition_matrix_by_day_minutely = pd.DataFrame(columns=aisles,
                                                     index=multi_idx)
    transition_matrix_by_day_minutely.columns.rename('next aisle',
                                                     inplace=True)

    # make a temporary version of the locations dataframe
    # this should have an extra column to avoid an indexing error later
    dummy_locations = get_customer_locations_by_time(data)
    dummy_locations["23:59:59"] = np.nan

    # loop through the aisles, find the table indices where that aisle appears
    # "next aisle" is the aisle named in the following column of the same row
    for day in day_idx:
        for a in aisles:
            if day == 'all':
                key = day_idx[1:]
            else:
                key = day
            instances_a = np.where(dummy_locations.loc[key] == a)
            instances_next = (instances_a[0], instances_a[1] + 1)
            counts = pd.Series(dummy_locations.loc[key]
                                              .values[instances_next])\
                       .value_counts()
            transition_matrix_by_day_minutely.loc[(day, a)] = counts

    # the total number of transitions from each aisle is the sum along the rows
    total_transitions_by_day = transition_matrix_by_day_minutely.sum(axis=1)

    # calculate the probability of each transition by dividing by the
    # total_transitions from each starting aisle
    transition_proba_by_day_minutely\
        = transition_matrix_by_day_minutely.div(total_transitions_by_day,
                                                axis=0)
    transition_proba_by_day_minutely\
        = transition_proba_by_day_minutely.fillna(0)
    return transition_proba_by_day_minutely


def joe_tm(customer_data):
    """
    wrapper to get a transition matrix for the average joe customer
    """
    df_location_by_time = get_customer_locations_by_time(customer_data)
    return get_trans_matrix(df_location_by_time)


def joe_ipmf(customer_data):
    """
    wrapper to get a n initial probability mass distribution for the average
    joe customer
    """
    df_location_by_time = get_customer_locations_by_time(customer_data)
    return get_first_aisle_pmf(df_location_by_time, day='all')
