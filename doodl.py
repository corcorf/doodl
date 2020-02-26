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


    def __init__(self, number, entry_time, initial_pmf, transition_matrix, exit_condition='checkout'):
        self.number = number
        self.entry_time = entry_time
        self.initial_pmf = initial_pmf
        self.transition_matrix = transition_matrix
        self.exit_condition = exit_condition
        self.history = []
        self._time_step = datetime.timedelta(0,60)
        self.__set_initial_state__()
        self.instantiated_at = datetime.datetime.now()


    def __repr__(self):
        return f"Customer {self.number}" #and history {', '.join(self.history)}

    def __str__(self):
        return f"Customer {self.number}, entry at {self.entry_time}, {self.initial_state}"

    def __set_initial_state__(self):
        """
        randomly selects an initial state from initial_pmf
        """
        self.initial_state  = choice(self.initial_pmf.index, 1, p=self.initial_pmf)[0]
        logging.debug(f"Initial state for customer with entry time {self.entry_time} is {self.initial_state}")

    def __get_initial_state__(self):
        return self.initial_state

#     @property
#     def initial_state(self):
#         return self.__initial_state

#     @initial_state.getter
#     def initial_state(self):
#         return self.__initial_state

#     @initial_state.setter
#     def initial_state(self):
# #         self.__initial_state = init
#         self.__initial_state  = choice(self.initial_pmf.index, 1, p=self.initial_pmf)[0]


    def go_shopping(self):
        time_elapsed = datetime.timedelta(0,0)

        current_time = self.entry_time
        current_state = self.__get_initial_state__()
#         current_state = self.__initial_state
        record = (current_time, current_state)
        self.history.append(record)
        yield record

        while current_state != self.exit_condition:
            current_time += self._time_step
            tm = self.transition_matrix.loc[current_state].dropna()
            current_state = choice(tm.index, 1, p=tm)[0]
            record = (current_time, current_state)
            self.history.append(record)
            yield record


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
    customer_counter (int): the number of customers that have passed through the supermarket

    _______
    methods:
    add_customer
    remove_customer
    track_customer
    move_time
    save_records

    """
    customers = []
    records = []
    customer_counter = 0

    def __init__(aisles, exit_state, customers=None,
        self, opening_time="09:00", closing_time="17:00",
        time_step=60, default_transition_matrix=None,
        default_initial_pmf=None,
    ):
        self.aisles = aisles
        self.exit_state = exit_state
        if customers != None:
            self.customers = customers
        self.opening_time = pd.to_datetime(opening_time).time()
        self.closing_time = pd.to_datetime(closing_time).time()
        self.time_step = datetime.timedelta(0,time_step)


    def __repr__():
        return "I am a SuperMarket"


    def add_customer(
    self, entry_time, number=None, transition_matrix=None, initial_pmf=None,
    ):
        """
        Add a customer to the supermarket
        """

        if not isinstance(transition_matrix, pd.DataFrame):
            if isinstance(self.default_transition_matrix, pd.DataFrame):
                transition_matrix = self.default_transition_matrix
            else:
                raise Exception('No transition matrix found.')

        if not isinstance(initial_pmf, pd.DataFrame):
            if isinstance(self.default_initial_pmf, pd.DataFrame):
                initial_pmf = self.default_initial_pmf
            else:
                raise Exception('No initial probability mass function found.')

        self.customers.append(
            Customer(
            self.customer_counter, entry_time, initial_pmf,
            transition_matrix, self.exit_state
            )
        )
        self.customer_counter += 1
