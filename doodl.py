import os
import pandas as pd
import numpy as np
from numpy.random import choice
import numpy.ma as ma
import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from imageio import imread, imwrite
from scipy.ndimage import rotate
from PIL import Image
import cv2

def load_data(path='data'):
    path = 'data'
    data = pd.DataFrame()
    for file in os.listdir(path):
        df = pd.read_csv(os.path.join(path,file), sep=';')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S")
        data = pd.concat([data,df])
    data = data.sort_values(by=['timestamp']).reset_index(drop=True)
    return data

def get_aisles():
    aisles = ['fruit', 'spices', 'dairy', 'drinks', 'checkout']
    return aisles

def get_first_aisle_pmf(df_locations, day='all'):
    """
    get probabilities for first aisle visited
    """
    aisles = get_aisles()
    first_aisle_proba = pd.DataFrame(index=aisles)

    if day=='all':
        first_aisle_proba['all days'] = df_locations.iloc[:,0].value_counts() / df_locations.iloc[:,0].count()
    else:
    # for day in df_locations.index.levels[0]:
        first_aisle_proba[f'day {day}'] = df_locations.loc[day].iloc[0].value_counts() / df_locations.loc[day].iloc[0].count()

    first_aisle_proba = first_aisle_proba.fillna(0).squeeze()
    return first_aisle_proba


def get_customer_locations_by_time(data):
    """
    """
    df_location_by_time = data.sort_values(by=['customer_no', 'timestamp'])
    df_location_by_time['day'] = df_location_by_time['timestamp'].dt.dayofweek
    entrance_time = df_location_by_time.groupby(['day', 'customer_no']).nth(0)['timestamp']
    df_location_by_time = df_location_by_time.set_index(['day', 'customer_no'])
    time_elapsed = df_location_by_time['timestamp'] - entrance_time
    df_location_by_time['time_elapsed'] = time_elapsed.sort_index(level=1).values ### THIS IS A BIT HAIRY - add unique index ###
    df_location_by_time = df_location_by_time.reset_index().set_index(['day', 'customer_no','time_elapsed'])['location']
    df_location_by_time = df_location_by_time.unstack(-1).reindex(pd.timedelta_range('00S', '3600S', freq='60S'), axis=1)
    df_location_by_time = df_location_by_time.fillna(method='ffill', axis=1)
    return df_location_by_time

def get_trans_matrix(df_location_by_time):
    """
    set up a dataframe for the transition matrix
    should have a column and a row for each aisle
    """
    aisles = get_aisles()
    transition_matrix_minutely = pd.DataFrame(columns=aisles, index=aisles[:-1])
    transition_matrix_minutely.index.rename('start aisle', inplace=True)
    transition_matrix_minutely.columns.rename('next aisle', inplace=True)

    # make a temporary version of the locations dataframe
    # this should have an extra column to avoid an indexing error later
    dummy_locations = df_location_by_time.copy()
    dummy_locations["23:59:59"] = np.nan

    # loop through the aisles, find the table indeces where that aisle appears
    # the "next aisle" is the aisle named in the following column of the same row
    for a in aisles:#[:-1]:
        instances_a = np.where(dummy_locations==a)
        instances_next = (instances_a[0], instances_a[1] + 1)
        transition_matrix_minutely.loc[a] = pd.Series(dummy_locations.values[instances_next]).value_counts()

    # the total number of transitions from each aisle is the sum along the rows
    total_transitions = transition_matrix_minutely.sum(axis=1)

    # calculate the probability of each transition by dividing by the total_transitions from each starting aisle
    transition_proba_minutely = transition_matrix_minutely.div(total_transitions,axis=0).fillna(0)
    return transition_proba_minutely

def get_trans_matrix_by_day(data, day='all'):
    """
    calculate the transition matrix by day
    set up a dataframe for the transition matrix
    should have a column and a row for each aisle
    """

    day_idx = ['all'] + list(range(5))
    multi_idx = pd.MultiIndex.from_product([day_idx,aisles], names=['day','start aisle'])
    transition_matrix_by_day_minutely = pd.DataFrame(columns=aisles, index=multi_idx)
    # transition_matrix_by_day_minutely.index.rename('start aisle', inplace=True)
    transition_matrix_by_day_minutely.columns.rename('next aisle', inplace=True)

    # make a temporary version of the locations dataframe
    # this should have an extra column to avoid an indexing error later
    dummy_locations = df_location_by_time.copy()
    dummy_locations["23:59:59"] = np.nan

    # loop through the aisles, find the table indices where that aisle appears
    # the "next aisle" is the aisle named in the following column of the same row
    for day in day_idx:
        for a in aisles:
            if day == 'all':
                key = day_idx[1:]
            else:
                key = day
            instances_a = np.where(dummy_locations.loc[key]==a)
            instances_next = (instances_a[0], instances_a[1] + 1)
            counts = pd.Series(dummy_locations.loc[key].values[instances_next]).value_counts()
            transition_matrix_by_day_minutely.loc[(day,a)] = counts


    # the total number of transitions from each aisle is the sum along the rows
    total_transitions_by_day = transition_matrix_by_day_minutely.sum(axis=1)

    # calculate the probability of each transition by dividing by the total_transitions from each starting aisle
    transition_proba_by_day_minutely = transition_matrix_by_day_minutely.div(total_transitions_by_day,axis=0)
    transition_proba_by_day_minutely = transition_proba_by_day_minutely.fillna(0)
    return transition_proba_by_day_minutely

def joe_tm():
    """
    wrapper to get a transition matrix for the average joe customer
    """
    data = load_data()
    df_location_by_time = get_customer_locations_by_time(data)
    return get_trans_matrix(df_location_by_time)

def joe_ipmf():
    """
    wrapper to get a n initial probability mass distribution for the average joe customer
    """
    data = load_data()
    df_location_by_time = get_customer_locations_by_time(data)
    return get_first_aisle_pmf(df_location_by_time, day='all')

class Customer:
    """
    Class representing a customer in the DOODL supermarket!

    Attributes:
        entry_time (datetime.datetime): the time at which the customer enters the supermarket
        initial_pmf (pandas.Series): probability mass function for the customer's initial state,
        i.e. which aisle the customer will go to first
        transition_matrix (pandas.DataFrame): transition matrix containing the probability of where the
        customer will head in the next minute, base on where they are now
        exit_state (string): the state at which the customer exits the simulation
    """


    def __init__(self, number, initial_pmf, transition_matrix):

        assert np.all(initial_pmf.index.isin(transition_matrix.index))
        self.initial_pmf = initial_pmf
        self.transition_matrix = transition_proba_minutely

        self.number = number
        self.__set_initial_state__()
        self.current_state = self.__get_initial_state__()


    def __repr__(self):
        return f"Customer {self.number}" #and history {', '.join(self.history)}

    def __str__(self):
        return f"Customer {self.number}" #", entry at {self.entry_time}, {self.initial_state}"

    def __set_initial_state__(self):
        """
        randomly selects an initial state from initial_pmf
        """
        self.initial_state  = choice(self.initial_pmf.index, 1, p=self.initial_pmf)[0]

    def __get_initial_state__(self):
        return self.initial_state

    @property
    def __record__(self):
        return self.current_state

    def __iter__(self):
        return self

    def __next__(self):
        record = self.__record__
        tm = self.transition_matrix.loc[self.current_state].dropna()
        self.current_state = choice(tm.index, 1, p=tm)[0]
        return record

standard_ipmf = joe_ipmf()
standard_tm = joe_tm()

class JoeCustomer(Customer):
    """
    Customer with initial_pmf and transition_matrix based on average of all customers
    """

    def __init__(self, number):

        self.initial_pmf = standard_ipmf
        self.transition_matrix = standard_tm

        assert np.all(self.initial_pmf.index.isin(self.transition_matrix.index))

        self.number = number
        self.__set_initial_state__()
        self.current_state = self.__get_initial_state__()


class SuperMarket:
    """
    Class representing a supermarket in which customers can shop

    _______
    params:
    aisles (list): list of aisle names in the supermarket
    exit_state (string): state at which customers should leave the supermarket
    customers (list): list of Customer objects to be included in the supermarket
    opening_time (string): time recognisable by pandas.to_datetime at which supermarket opens
    closing_time (string): time recognisable by pandas.to_datetime at which supermarket closes
    time_step (int): time step in seconds (default 60 seconds)
    default_transition_matrix (pandas.DataFrame):
    default_initial_pmf (pandas.Series):

    _______
    attributes:
    records (...): movement records for all customers that have been in the supermarket
    turnstile_counter (int): the number of customers that have passed through the supermarket

    _______
    methods:
    add_customer
    remove_customer
    track_customer
    move_time
    save_records

    """
    customers = {}
    turnstile_counter = 0
    records_df = pd.DataFrame()

    at_checkout = []
    checkout_records = pd.DataFrame()
    queuing_times = pd.DataFrame()

    def __init__(
        self, aisles, n_checkouts=1, checkout_rate=0.5,
        exit_state="checkout", customers=None,
        opening_time="09:00", closing_time="17:00", time_step=60,
        # default_transition_matrix=None,
        # default_initial_pmf=None,
    ):
        self.aisles = aisles
        self.exit_state = exit_state
        if customers != None:
            self.customers = customers
        self.opening_time = pd.to_datetime(opening_time).time()
        self.closing_time = pd.to_datetime(closing_time).time()
        self.time_step = datetime.timedelta(0,time_step)

        assert n_checkouts > 0 and checkout_rate > 0
        self.n_checkouts = n_checkouts
        self.checkout_rate = checkout_rate


    def __repr__():
        return "I am a SuperMarket"

    @property
    def customer_type(self):
        """
        Select the customer class to choose based on time and day
        """
        ...

        return JoeCustomer

    @property
    def n_new_customers(self):
        """
        Choose the number of new customers that should be added in the current timestep
        """
        ...
        return 2

    def add_customers(self):
        """
        Add new customers to the supermarket
        """
        for n in range(self.n_new_customers):

            self.customers[self.turnstile_counter] = self.customer_type(self.turnstile_counter)
            self.queuing_times.loc[self.turnstile_counter,"time"] = self.time_step * 0
            self.turnstile_counter += 1



    def yield_customer_locations(self, customer_list):
        """
        Get the new locations for every customer in the supermarket
        """
        for i, c in customer_list.items():
            try:
                yield np.array([(i, next(c))], dtype=[('customer_no', 'int32'), ('location', 'U10')])
            except StopIteration:
                pass

    def append_new_records(self, time):
        """
        Get customer records for the current timestep and append them to the supermarket records df
        """
        recs = self.yield_customer_locations(self.customers)

        self.new_records = np.array([r for r in recs]).reshape(-1)
        self.new_records = pd.DataFrame(self.new_records, index=range(len(self.new_records)))
        self.new_records['timestamp'] = time

        self.records_df = self.records_df.append(self.new_records, ignore_index=True)

    def update_checkout(self):
        """check the new records for any customers who have reached the checkout area"""
        new_at_checkout = self.new_records.loc[
            self.new_records['location']==self.exit_state,
            "customer_no"
        ].tolist()
        new_at_checkout = [n for n in new_at_checkout if n not in self.at_checkout]
        logging.debug(f"new to checkout {new_at_checkout}")
        self.at_checkout += new_at_checkout

    def work_checkout(self):
        """
        Operate the checkouts so the good people can go home and get on with their lives.
        If self.checkout_rate < 1, use a Binomial distribution with trial number equal to
        the number of open checkouts to sample the number of customers that get to leave
        each minute.
        If self.checkout_rate >= 1, that number of customers are processed from each checkout
        every minute
        """
        if self.checkout_rate < 1:
            max_processed = np.random.binomial(self.n_checkouts, self.checkout_rate)
        else:
            max_processed = self.checkout_rate
        logging.debug(f"Max of {max_processed} customers may be checked out")

        leaving = []
        n_processed = 0
        logging.debug(f"currently at checkout: {self.at_checkout}")

        for cust in self.at_checkout:
            if n_processed >= max_processed:
                break
            leaving.append(cust)
            self.customers.pop(cust) # could use None as second argument
            n_processed += 1

        logging.debug(f"{n_processed} customers checked out: {leaving}")
        for cust in leaving:
            self.at_checkout.remove(cust)


    def update_queue_records(self, time):
        self.queuing_times.loc[self.at_checkout,'time'] += self.time_step
        self.checkout_records.loc[time,"queue_length"] = len(self.at_checkout)

    def add_date_to_records(self, date):
        self.records_df['datetime'] = self.records_df['timestamp'] + date

    @property
    def time_range(self):
        """Get a pandas timedelta_range object for the supermarket's opening times and timestep"""
        ot = datetime.timedelta(days=self.opening_time.hour*1/24, seconds=self.opening_time.minute*60)
        ct = datetime.timedelta(days=self.closing_time.hour*1/24, seconds=self.closing_time.minute*60)
        return pd.timedelta_range(ot, ct, freq=self.time_step)

    def get_new_customers(self):
        self.add_customers()

    def day_in_the_life(self, date=None):
        """
        Simulate a day's shopping

        Return a dataframe with the day's customer records
        """

        for time in self.time_range:
            logging.debug(f"Time is {time}")

            self.get_new_customers()
            self.append_new_records(time)
            self.update_checkout()
            self.work_checkout()
            self.update_queue_records(time)

        if date != None:
            self.add_date_to_records(date)

        return self.records_df

    ######## visualisation stuff ############
    def set_visualisation_params(self):
        """Set up the parameters for the visualisation function"""
        self.img = cv2.imread('market.png')
        self.img_height, self.img_width = self.img.shape[:2]

        self.aisle_colour = [200,0,200]
        self.checkout_colour = [255, 0, 0]
        self.square_side = 20
        self.offset = 80
        self.no_value = -999

        # parameters for customer icons in aisles
        self.ncol_aisles = 2
        self.nrow_aisles = 10
        self.aisle_icon_divide = 20
        self.aisles_top = 150

        # parameters for customer icons in checkouts
        self.cols_per_checkout = 2
        self.ncol_checkouts = self.n_checkouts * 2
        self.nrow_checkouts = 23
        self.checkout_icon_divide = 7
        self.checkout_bottom = 600
        self.checkout_section_width = 350

    def set_up_visualisation_matrix_for_aisles(self):
        """Set up the image masks for the aisle customer icons"""
        for i,a in enumerate(self.aisles[:-1]):
            self.visualisation_matrices[a] = np.ones_like(self.img) * self.no_value
            for loc in range(self.nrow_aisles * self.ncol_aisles):
                row = loc // self.ncol_aisles
                col = loc % self.ncol_aisles
                yul = self.aisles_top + (self.square_side + self.aisle_icon_divide) * row
                xul = self.offset + int(self.img_width/4) * i + (self.square_side + self.aisle_icon_divide) * col

                self.visualisation_matrices[a][yul:yul+self.square_side, xul:xul+self.square_side] = loc

            mask = self.visualisation_matrices[a]==self.no_value
            self.visualisation_matrices[a] = ma.masked_array(self.visualisation_matrices[a], mask=mask)


    def set_up_visualisation_matrix_for_checkouts(self):
        """Set up the image masks for the checkout customer icons"""
        self.visualisation_matrices['checkout'] = np.ones_like(self.img) * self.no_value
        for loc in range(self.nrow_checkouts * self.ncol_checkouts):
            row = loc // self.ncol_checkouts
            col = loc % self.ncol_checkouts
            yul = self.checkout_bottom - (self.square_side + self.checkout_icon_divide) * row
            xul = (self.offset + int(self.checkout_section_width/4)
                                * (col//self.cols_per_checkout)
                                + (self.square_side + self.checkout_icon_divide) * col)
            self.visualisation_matrices['checkout'][yul:yul+self.square_side,
                                                    xul:xul+self.square_side] = loc+1

        mask = self.visualisation_matrices['checkout']==self.no_value
        self.visualisation_matrices['checkout'] = ma.masked_array(self.visualisation_matrices['checkout'], mask=mask)

    def set_up_visualisation_matrices(self):
        """Set up the matrices that will be used to add customer icons to the visualisation"""

        self.visualisation_matrices = {}

        self.set_up_visualisation_matrix_for_aisles()
        self.set_up_visualisation_matrix_for_checkouts()

    def loop_frames(self, df):
        # arguments for cv2 text
        org = (80,100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        text_colour = (0, 0, 0)
        thickness = 2

        for time, data in df.iterrows():
            frame = self.img.copy()

            for aisle, ncustomers in data.iteritems():
                mask = ma.where((self.visualisation_matrices[aisle]<ncustomers)[:,:,0])
                frame[mask] = self.aisle_colour

            ss = time.seconds
            time_string = "Time: {:02d}:{:02d}".format(ss//3600,(ss-(ss//3600)*3600)//60)
            frame = cv2.putText(frame, time_string, org, font,
                           fontScale, text_colour, thickness, cv2.LINE_AA)

            cv2.imshow('frame', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def visualise(self):
        df = self.records_df.groupby(['timestamp', "location"])['customer_no'].count().unstack(-1).fillna(0).astype(int)

        self.set_visualisation_params()
        self.set_up_visualisation_matrices()
        self.loop_frames(df)
