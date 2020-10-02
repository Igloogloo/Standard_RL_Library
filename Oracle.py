import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn

# The following is a general purpose Oracle used to simulate human feedback for reinforcement learning tasks.

class Oracle:
    def __init__(self, qtable=None):  # Takes in a qtable in pandas dataframe format. (Can easily convert from an
        # np array)
        # This implementation assumes states are represented by columns and actions by rows.
        self.qtable = qtable

    # The following is a simple oracle which returns feedback of 1 if the action is optimal and -1 otherwise
    def get_binary_feedback(self, state, action):
        optimal_action = self.qtable.loc[state].idxmax()
        if self.qtable.loc[state][optimal_action] == self.qtable.loc[state][action]:
            return 1
        else:
            return -1

    # The following function simulates a human by, given a state and an action already taken , returning binary feedback
    # The ps stands for policy shaping.
    # The parameter liklihd stands for likelihood and is the probability that the oracle will provide feedback.
    # The parameter conf is the confidence that the oracle's feedback is optimal.
    def get_binary_feedback_ps(self, state, action, liklihd, conf):
        if liklihd < 0 or liklihd > 1:
            raise ValueError('Likelihood parameter must be a value between 0 and 1')

        if np.random.random() > liklihd:  # If True, 0 feedback is returned
            return 0

        if conf < 0 or conf > 1:
            raise ValueError('Confidence parameter must be a value between 0 and 1')

        if conf > np.random.random():  # If True, oracle will provide feedback optimally 
            optimal_action = self.qtable.loc[state].idxmax()
            if self.qtable.loc[state][optimal_action] == self.qtable.loc[state][action]:
                return 1
            else:
                return -1
        else:
            optimal_action = self.qtable.loc[state].idxmax()
            if self.qtable.loc[state][optimal_action] == self.qtable.loc[state][action]:
                return -1
            else:
                return 1

    def get_qval(self, state, action):
        return self.qtable[state][action]


class NPOracle:
    def __init__(self, qtable=None):  # Takes in a qtable as a numpy array.
        # This implementation assumes states are represented by columns and actions by rows.
        self.qtable = qtable

    # The following is a simple oracle which returns feedback of 1 if the action is optimal and -1 otherwise
    def get_binary_feedback(self, state, action):
        optimal_action = np.argmax(self.qtable[state])
        if self.qtable[state][optimal_action] == self.qtable[state][action]:
            return 1
        else:
            return -1

    # The following function simulates a human by, given a state and an action already taken , returning binary feedback
    # The ps stands for policy shaping.
    # The parameter liklihd stands for likelihood and is the probability that the oracle will provide feedback.
    # The parameter conf is the confidence that the oracle's feedback is optimal.
    def get_binary_feedback_ps(self, state, action, liklihd, conf):
        if liklihd < 0 or liklihd > 1:
            raise ValueError('Likelihood parameter must be a value between 0 and 1')

        if np.random.random() > liklihd:  # If True, 0 feedback is returned
            return 0

        if conf < 0 or conf > 1:
            raise ValueError('Confidence parameter must be a value between 0 and 1')

        if conf > np.random.random():  # If True, oracle will provide feedback optimally
            optimal_action = np.argmax(self.qtable[state])
            if self.qtable[state][optimal_action] == self.qtable[state][action]:
                return 1
            else:
                return -1

        else:
            optimal_action = np.argmax(self.qtable[state])
            if self.qtable[state][optimal_action] == self.qtable[state][action]:
                return -1
            else:
                return 1

    def get_qval(self, state, action):
        return self.qtable[state][action]

    
class TorchOracle:
    # Pytorch oravle class. Create an oracle with a pretrained model.
    # The "state" parameter is the input the oracle will evaluate. The action
    # parameter is the action that was taken by the agent that the oracle 
    # is evaluating against. 
    # All parameters are assumed to be pytorch tensors.
    def __init__(self, model):
        self.model = model
        model.eval()
    
    # The following is a simple oracle which returns feedback of 1 if the action is optimal and -1 otherwise
    def get_binary_feedback(self, state, action):
        with torch.no_grad():
            output = self.model(state)
            optimal_action = torch.argmax(output)
            optimal_action = optimal_action.item()
            if optimal_action == action.item():
                return 1
            else: 
                return -1
    
    # The following function simulates a human by, given a state and an action already taken , returning binary feedback
    # The ps stands for policy shaping.
    # The parameter liklihd stands for likelihood and is the probability that the oracle will provide feedback.
    # The parameter conf is the confidence that the oracle's feedback is optimal.
    def get_binary_feedback_ps(self, state, action, liklihd, conf):
        if liklihd < 0 or liklihd > 1:
            raise ValueError('Likelihood parameter must be a value between 0 and 1')

        if np.random.random() > liklihd:  # If True, 0 feedback is returned
            return 0

        if conf < 0 or conf > 1:
            raise ValueError('Confidence parameter must be a value between 0 and 1')
        
        with torch.no_grad():
            output = self.model(state)
            optimal_action = torch.argmax(output)
            optimal_action = optimal_action.item()
            print(f'optimal: {optimal_action}, action: {action.item()}')
            if conf > np.random.random():  # If True, oracle will provide feedback optimally
                if optimal_action == action.item():
                    return 1
                else: 
                    return -1
            else: 
                if optimal_action == action.item():
                    return -1
                else:
                    return 1
