import os
import pandas as pd
import numpy as np
import datetime
from numpy.random import choice
import logging
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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

class JoeCustomer(Customer):
    """
    Customer with initial_pmf and transition_matrix based on average of all customers
    """


    def __init__(self, number):

        self.initial_pmf = initial_choice_proba
        self.transition_matrix = transition_matrix

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
        default_transition_matrix=None,
        default_initial_pmf=None,
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


    def choose_customer_type(self, time, day):
        """
        Select the customer class to choose based on time and day
        """
#         if not isinstance(transition_matrix, pd.DataFrame):
#             if isinstance(self.default_transition_matrix, pd.DataFrame):
#                 transition_matrix = self.default_transition_matrix
#             else:
#                 raise Exception('No transition matrix found.')

#         if not isinstance(initial_pmf, pd.DataFrame):
#             if isinstance(self.default_initial_pmf, pd.DataFrame):
#                 initial_pmf = self.default_initial_pmf
#             else:
#                 raise Exception('No initial probability mass function found.')

        ...

        return JoeCustomer

    def add_customers(self, n_customers, customer_class=JoeCustomer,
                      transition_matrix=None, initial_pmf=None,
    ):
        """
        Add n new customers to the supermarket
        """

        for n in range(n_customers):

            self.customers[self.turnstile_counter] = customer_class(self.turnstile_counter)
            self.queuing_times.loc[self.turnstile_counter,"time"] = self.time_step * 0
            self.turnstile_counter += 1

    def get_n_new_customers(self):
        """
        Choose the number of new customers that should be added in the current timestep
        """
        ...
        return 2

    def yield_customer_locations(self, customer_list):
        """
        Get the new locations for every customer in the supermarket
        """
        for i, c in customer_list.items():
            try:
                yield np.array([(i, next(c))], dtype=[('customer_no', 'int32'), ('location', 'U10')])
            except StopIteration:
                pass

    def append_records(self, time):
        """
        Get customer records for the current timestep and append them to the supermarket records df
        """
        recs = self.yield_customer_locations(self.customers)

        self.new_records = np.array([r for r in recs]).reshape(-1)
        self.new_records = pd.DataFrame(self.new_records, index=range(len(self.new_records)))
        self.new_records['timestamp'] = time

        self.records_df = self.records_df.append(self.new_records, ignore_index=True)

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

        self.queuing_times.loc[self.at_checkout,'time'] += self.time_step


    def day_in_the_life(self, date):
        """
        Simulate a day's shopping

        Return a dataframe with the day's customer records
        """
        # initialise records and containers
        checked_out = np.array([-999])

        ot = datetime.timedelta(days=self.opening_time.hour*1/24, seconds=self.opening_time.minute*60)
        ct = datetime.timedelta(days=self.closing_time.hour*1/24, seconds=self.closing_time.minute*60)
        time_range = pd.timedelta_range(ot, ct, freq=self.time_step)


        for time in time_range:
            logging.debug(f"Time is {time}")

            n_new_customers = self.get_n_new_customers()
            customer_type = self.choose_customer_type(time, date.dayofweek)

            self.add_customers(n_new_customers, customer_type)

            self.append_records(time)

            new_at_checkout = self.new_records.loc[self.new_records['location']==exit_state,
                                                   "customer_no"].tolist()
            new_at_checkout = [n for n in new_at_checkout if n not in self.at_checkout]
            logging.debug(f"new to checkout {new_at_checkout}")

            self.at_checkout += new_at_checkout
            self.work_checkout()

            self.checkout_records.loc[time,"queue_length"] = len(self.at_checkout)

        ### maybe call this datetime to avoid confusion with the dateless version?
        self.records_df['datetime'] = self.records_df['timestamp'] + date

        return records_df
