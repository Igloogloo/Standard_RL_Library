import numpy as np


# The following is a class that implements Policy Shaping given a q-table, a feedback table and the necessary parameters
# The primary function of the class is to easily compute the next action to take in a given state according to the
# Policy Shaping algorithm

class PolicyShaping:
    def __init__(self, qtable, feedback_tbl, confidence, constant):
        """
        :param constant: Constant for Boltzmann Exploration
        """
        self.qtable = qtable
        self.feedback_tbl = feedback_tbl
        self.confidence = confidence
        self.constant = constant
        self.col_size, self.action_size = qtable.shape

    def update_qtable(self, qtable):  # Update after each change to the q-table.
        self.qtable = qtable

    def update_feedback_tbl(self, feedback_tbl):  # Update after each change to the feedback table.
        self.feedback_tbl = feedback_tbl

    def set_confidence(self, confidence):
        self.confidence = confidence

    def set_constant(self, constant):
        self.constant = constant

    def get_shaped_action(self, state, confidence=None, constant=None):
        # Returns the action that results from the policy shaping algorithm applied for the given state.
        if confidence is None:
            confidence = self.confidence
        if constant is None:
            constant = self.constant

        prob_of_actions = []  # Probability distribution over actions in the current.
        for action in range(self.action_size):  # Calculate the probability of each action given the current state.
            p_act = p_action(self.qtable, state, action, constant)
            p_gd = p_good(self.feedback_tbl, confidence, state, action)
            denominator = 0
            for other_action in range(self.action_size):
                denominator += (p_action(self.qtable, state, other_action, constant) *
                                p_good(self.feedback_tbl, confidence, state, other_action))
            prob_of_actions.append(shaped_policy(p_act, p_gd, denominator))
        action = int(np.random.choice([i for i in range(self.action_size)], p=prob_of_actions))
        return action


def p_action(qtable, state, action, const):
    """
    Returns the probability of taking an action given the learned q-values
    for the given state, using Boltzmann Exploration
    """
    numerator = np.exp(qtable[state][action] / const)

    modified_actions = np.copy(qtable[state])
    for i in range(len(modified_actions)):
        modified_actions[i] = np.exp(modified_actions[i] / const)

    denominator = np.sum(modified_actions)

    return numerator / denominator


def p_good(feedback_tbl, confidence, state, action):
    """
    Returns the probability that an action is optimal based off of previous feedback and given the confidence in the
    feedback.
    """
    # The following is a pretty arbitrary way to prevent overflow, it also artificially limits how much feedback
    # affects the agent. Worth changing if you know a better way.
    if (feedback_tbl[state][action]) > 100:
        return 1
    elif (feedback_tbl[state][action]) < -100:
        return 0

    if confidence == 1:
        confidence == .9999999

    return (np.power(confidence, feedback_tbl[state][action])) / \
           (np.power(confidence, feedback_tbl[state][action]) + np.power(1 - confidence,
                                                                         feedback_tbl[state][action]))


def shaped_policy(p_action, p_good, denominator):
    """
    Returns the probability of an action given a state for the policy as a result of policy shaping
    :param denominator: The sum over all p_action and p_good for every action in a given state.
    """

    return (p_action * p_good) / denominator
