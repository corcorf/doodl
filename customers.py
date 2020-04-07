"""
Module defining customer classes
"""

import numpy as np
from numpy.random import choice
from data_processing import load_data, joe_ipmf, joe_tm

CUSTOMER_DATA = load_data()
JOE_IPMF = joe_ipmf(CUSTOMER_DATA)
JOE_TM = joe_tm(CUSTOMER_DATA)


class Customer:
    """
    Class representing a customer in the DOODL supermarket!

    Attributes:
        entry_time (datetime.datetime): the time at which the customer enters
                                        the supermarket
        initial_pmf (pandas.Series): probability mass function for the
                                     customer's initial state,
        i.e. which aisle the customer will go to first
        transition_matrix (pandas.DataFrame): transition matrix containing the
                                              probability of where the
        customer will head in the next minute, base on where they are now
        exit_state (string): the state at which the customer exits the
                             simulation
    """

    def __init__(self, number, initial_pmf, transition_matrix):
        assert np.all(initial_pmf.index.isin(transition_matrix.index))
        self.initial_pmf = initial_pmf
        self.transition_matrix = transition_matrix

        self.number = number
        self.__set_initial_state__()
        self.current_state = self.__get_initial_state__()

    def __repr__(self):
        return f"Customer {self.number}"

    def __str__(self):
        return f"Customer {self.number}"

    def __set_initial_state__(self):
        """
        randomly selects an initial state from initial_pmf
        """
        self.initial_state = choice(
                                self.initial_pmf.index,
                                1,
                                p=self.initial_pmf
                            )[0]

    def __get_initial_state__(self):
        """
        Return the customer's initial state
        """
        return self.initial_state

    @property
    def __record__(self):
        """
        Return the customer's state record
        """
        return self.current_state

    def __iter__(self):
        """
        Initialise generator for the customer states
        """
        return self

    def __next__(self):
        """
        Get next customer state
        """
        record = self.__record__
        tm = self.transition_matrix.loc[self.current_state].dropna()
        self.current_state = choice(tm.index, 1, p=tm)[0]
        return record


class JoeCustomer(Customer):
    """
    Customer with initial_pmf and transition_matrix based on average of all
    customers
    """

    def __init__(self, number):
        self.initial_pmf = JOE_IPMF
        self.transition_matrix = JOE_TM
        assert\
            np.all(self.initial_pmf.index.isin(self.transition_matrix.index))
        self.number = number
        self.__set_initial_state__()
        self.current_state = self.__get_initial_state__()
