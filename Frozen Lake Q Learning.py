""""
The following is a finite q-learning example in the openai gym env. Frozen lake
(see: https://gym.openai.com/envs/FrozenLake-v0/)

The basis of this code was taken from DeepLizard. See DeepLizard for an excellent breakdown of q-learning and
an in depth look at the code: https://deeplizard.com/learn/video/qhRNvCVVJaA

"""

import gym
import numpy as np
import Qtable
import matplotlib.pyplot as plt
from PolicyShaping import p_action

# is_slippery refers to weather or not the agent will take the action chosen with 100% success rate

# Make environment, size: 8x8
#lake = gym.make("FrozenLake-v0", map_name='8x8', is_slippery=False)

# Make enviornment, size 4x4
lake = gym.make("FrozenLake-v0", is_slippery=True)

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
qtable = Qtable.Qtable(state_size, action_size, True)  # Initialize qtable

episodes = 10000
max_moves = 100  # Max moves per episode

# Initialize parameters:
learning = .1
discount = .99
exploration_rate = 1
min_exploration = .01
exploration_decay = .001

rewards = []
num_moves = []

for episode in range(episodes):
    state = lake.reset()  # It is crucial to reset OpenAI gym enviornments at the start of each episode
    done = False
    curr_reward = 0
    num_moves.append(0)

    for move in range(max_moves):

        # Use epsilon greedy to take optimal action or choose action randomly
        if np.random.random() > exploration_rate:
            action = qtable.optaction(state)
            '''
            # Uncomment if you want the agent to take actions probabilistically instead of greedily
            act_probs = []
            for act in range(action_size):
                act_probs.append(p_action(qtable.qtable, state, act, const=.5))
            action = np.random.choice([i for i in range(action_size)], p=act_probs)
            '''

        else:
            action = lake.action_space.sample()

        num_moves[episode] += 1

        # For all OpenAI gym enviornments there is a similar "step" function
        next_state, reward, done, info = lake.step(action)

        # if episode%100 == 0 and move == 0:
        #    print(action)

        # You can customize the reward function with some code like that below:
        """
        # Give a minus 1 reward for the episode finishing but not being at the goal:
        if reward == 0 and done is True: 
            reward = -1
            
        # Give a minus 1 reward for the agent taking action 2
        if reward == 0 and action == 2:
            reward = -1
        """

        # Q-value update formula
        qtable.qtable[state][action] = (1 - learning) * qtable.qtable[state][action] + learning * \
                                       (reward + discount * qtable.maxq(next_state))

        state = next_state
        curr_reward += reward

        if done:
            break

    rewards.append(curr_reward)

    # Exponential decay of exploration rate:
    exploration_rate = min_exploration + (1 - min_exploration) * np.exp(-exploration_decay * episode)

# Print average rewards over the episodes:
rewards_per_thosand_episodes = np.split(np.array(rewards), episodes / 1000)
count = 1000
avg_rewards = []
for r in rewards_per_thosand_episodes:
    avg_reward = sum(r / 1000)
    avg_rewards.append(avg_reward)
    print(count, ": ", str(avg_reward))
    count += 1000

episodes = np.arange(0, 10000, 1000)

# Plot average rewards
plt.plot(episodes, avg_rewards)
plt.show()