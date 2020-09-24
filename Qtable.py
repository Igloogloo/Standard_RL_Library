import numpy as np
import pandas as pd
import pickle


class Qtable:
    def __init__(self, state_size, action_size, zeros=True, minqval=None, maxqval=None, qtable=None):
        """
        If zeros is set to False, then the q-table will be initiated with random values between 0 and 1. If min and
        max values are set, the table is filled with uniform distributed values between the min and max.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.minqval = minqval
        self.maxqval = maxqval
        if qtable is not None:
            self.qtable = qtable
        else:
            if zeros is True:
                self.qtable = np.zeros((state_size, action_size))

            elif minqval is None and maxqval is None and zeros is False:
                self.qtable = np.random.random((state_size, action_size))

            else:
                self.qtable = np.random.uniform(minqval, maxqval, (state_size, action_size))

    def get_qtable(self):
        return self.qtable

    def set_qtable(self, qtable):
        self.qtable = qtable

    def get_qval(self, state, action):
        """
        Returns the q-vlaue given a state acion pair.
        """
        try:
            return self.qtable[state, action]
        except IndexError:
            if action >= self.action_size:
                print("Index error: No such action.")

            elif state >= self.state_size:
                print("Index error: No such state.")

            elif state >= self.state_size and action >= self.action_size:
                print("Index error: No such state nor such action.")

    def set_qval(self, state, action, qval):
        """
        Assign a q-value to a state action pair.
        """
        try:
            self.qtable[state, action] = qval
        except IndexError:
            if action >= self.action_size:
                print("Index error: No such action.")
            elif state >= self.state_size:
                print("Index error: No such state.")
            elif state >= self.state_size and action >= self.action_size:
                print("Index error: No such state nor such action.")

    def maxq(self, state):
        """
        Returns the maximum q-value for a given state.
        """
        try:
            return np.max(self.qtable[state])
        except IndexError:
            print("Index error: No such state.")

    def optaction(self, state):
        """
        Returns the action associated with the highest q-value given a state
        """
        try:
            return np.argmax(self.qtable[state])
        except IndexError:
            print("Index error: No such state.")

    def random_action(self, state):
        """
        Returns a random action to be taken for a given state
        """
        try:
            return np.random.randint(0, self.action_size)
        except IndexError:
            print("Index error: No such state.")

    def optsequence(self):
        """
        Returns the sequence of actions associated with the highest q-values for each state in  order from 0-state_siz
        """
        return np.argmax(self.qtable, axis=1)

    def save_qtable(self, episode):
        """
        Save qtable object in pickle file.
        """
        with open(f'q_table_episode_{episode}.pkl', 'wb') as pkl:
            pickle.dump(self, pkl)
            pkl.close()
