"""
The following is a discrete policy shaping example in the openAI Frozen lake text env.
(see: https://gym.openai.com/envs/FrozenLake-v0/)

For more details on policy shaping see both the PolicyShaping class and the original paper
by  Griffith (see: https://papers.nips.cc/paper/2013/file/e034fb6b66aacc1d48f445ddfb08da98-Paper.pdf)
"""
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import Qtable
from Oracle import NPOracle as Oracle
import PolicyShaping

# is_slippery refers to weather or not the agent will take the action chosen with 100% success rate

# Make environment, size: 8x8
#lake = gym.make("FrozenLake-v0", map_name='8x8', is_slippery=False)

# Make enviornment, size 4x4
lake = gym.make("FrozenLake-v0", is_slippery=False)

# How to make a custom map:
"""
cstm_map = [
    'FFSFF',
    'FHHHF',
    'FHHHF',
    'FHHHF',
    'GFFFG'
]

lake = gym.make("FrozenLake-v0", desc=cstm_map, is_slippery=False)
"""

state_size = lake.observation_space.n
action_size = lake.action_space.n
qtable = Qtable.Qtable(state_size, action_size, True)  # Initialize q-table

# Initialize oracle
oracle_qtable_initial = pickle.load(open('q_table_episode_4x4Oracle.pkl', 'rb')).qtable
oracle = Oracle(oracle_qtable_initial)
print(oracle.qtable)

episodes = 100
max_moves = 30  # Max moves per episode

# q-learning parameters
learning = .8
discount = .99
exploration_rate = .001
min_exploration = .01
exploration_decay = .001

# policy shaping parameters:
confidence = .9  # confidence that feedback is optimal
likelihood = .9  # likelihood feedback is provided
const = 0.3  # constant used in probability of action equation
feedback_tbl = np.zeros((state_size, action_size))  # Table keeping track of feedback

policy_shaping = PolicyShaping.PolicyShaping(qtable.qtable, feedback_tbl, confidence, const)

rewards = []
num_moves = []

for episode in range(episodes):
    state = lake.reset()
    done = False
    curr_reward = 0
    num_moves.append(0)

    for move in range(max_moves):
        # This epsilon-greedy approach in PS with QL is analogous to the original papers, exploration rate is
        # a const depending on the env. they used.
        if np.random.random() > exploration_rate:
            action = policy_shaping.get_shaped_action(state)
        else:
            action = lake.action_space.sample()

        num_moves[episode] += 1
        next_state, reward, done, info = lake.step(action)

        """
        if reward == 0 and done is True:  # Modifying frozen lake so falling in a hole gives - 1 reward instead of 0.
            reward = -1
        """

        # Get feedback and update feedback table:
        feedback = oracle.get_binary_feedback_ps(state, action, likelihood, confidence)
        feedback_tbl[state][action] += feedback

        # Q-value update formula
        qtable.qtable[state][action] = (1 - learning) * qtable.qtable[state][action] + learning * \
                                       (reward + discount * qtable.maxq(next_state))

        # Update policy shaping object
        policy_shaping.update_qtable(qtable.qtable)
        policy_shaping.update_feedback_tbl(feedback_tbl)

        state = next_state
        curr_reward += reward

        if done:
            break

    rewards.append(curr_reward)

    """
    # Exponential decay of exploration rate:
    exploration_rate = min_exploration + (1 - min_exploration) * np.exp(-exploration_decay * episode)
    """

# print(qtable.qtable)

rewards_per_thosand_episodes = np.split(np.array(rewards), episodes / 10)
count = 10
avg = []
for r in rewards_per_thosand_episodes:
    avg_reward = sum(r / 10)
    print(count, ": ", str(avg_reward))
    avg.append(avg_reward)
    count += 10

plt.plot(avg)
plt.show()