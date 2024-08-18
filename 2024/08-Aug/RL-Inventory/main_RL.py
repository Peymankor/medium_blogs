import numpy as np
import pandas as pd
from scipy.stats import poisson
import random

class QLearningInventory:
    def __init__(self, user_capacity, poisson_lambda, holding_cost, stockout_cost, 
                 gamma, alpha, epsilon, episodes, max_actions_per_episode):
        # Initialize the parameters for the Q-learning algorithm
        self.user_capacity = user_capacity
        self.poisson_lambda = poisson_lambda
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.gamma = gamma  # Discount factor
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.episodes = episodes  # Number of episodes
        self.Q = self.initialize_Q()  # Initialize the Q-table
        self.max_actions_per_episode = max_actions_per_episode  # Maximum number of actions per episode

    def initialize_Q(self):
       
       # Initialize the Q-table as a dictionary
        Q = {}
    
        # alpha is the number of items in stock (On-Hand Inventory)
        # beta is the number of items on order (On-Order Inventory)
        for alpha in range(self.user_capacity + 1):
            for beta in range(self.user_capacity + 1 - alpha):
                
                state = (alpha, beta)
            # Initialize the nested dictionary for each state
                Q[state] = {}
            # Determine the possible actions based on the current state
                max_action = self.user_capacity - (alpha + beta)
                for action in range(max_action + 1):
                    
                    Q[state][action] = 0 
                    #Q[state][action] = np.random.uniform(0, 1)  # Start with small random values instead of 0
    
        return Q

    def get_next_state_and_reward(self, state, action):
        
        # Simulate the environment to get the next state and reward
        alpha, beta = state  # alpha is the current inventory level
        init_inv = alpha + beta  # beta is the number of items on order

        demand = np.random.poisson(self.poisson_lambda)
        #print("Demand:", demand)
        
        # Calculate the new inventory level after demand is realized
        new_alpha = max(0, init_inv - demand)
        
        # Calculate the reward based on holding costs for remaining inventory
        holding_cost = -new_alpha * self.holding_cost

        stockout_cost = 0

        if demand > init_inv:
            stockout_cost = -(demand - init_inv) * self.stockout_cost
    
        reward = holding_cost + stockout_cost
        
        # Define the next state as the remaining inventory and the action taken
        
        next_state = (new_alpha, action)

        return next_state, reward
        
        # Simulate the environment to get the next state and reward
        
        # alpha, beta = state
        # init_inv = alpha + beta
        # #new_alpha = init_inv + action
        # demand = np.random.poisson(self.poisson_lambda)
        # print("Demand:", demand)
        # if demand <= (init_inv-1):
        #     reward = -alpha * self.holding_cost
        #     new_alpha = init_inv - demand
        # else:
        #     transition_prob = 1 - poisson.cdf(init_inv - 1, self.poisson_lambda)
        #     transition_prob2 = 1 - poisson.cdf(init_inv, self.poisson_lambda)
        #     reward = -alpha * self.holding_cost - self.stockout_cost * (
        #         (self.poisson_lambda * transition_prob) - init_inv * transition_prob2)
            
        #     new_alpha = 0
        
        # next_state = (new_alpha, action)
        
        # return next_state, reward

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.user_capacity - 
                                    (state[0] + state[1]) + 1)  # Explore
        else:

            best_action= max(self.Q[state], key=self.Q[state].get)
            
            return best_action
            #return np.argmax(self.Q[state])  # Exploit

    def update_Q(self, state, action, reward, next_state):
        # Update the Q-table based on the state, action, reward, and next state
        best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)
        np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train(self):
        # Train the Q-learning algorithm over the specified number of episodes
        rewards_per_episode = []
        #cumulative_rewards = 0

        for episode in range(self.episodes):
            
            state = (0, 0)
            #state = (0, 0)  # Initialize the state at the start of each episode
            total_reward = 0  # Initialize the total reward for the current episode
            action_taken = 0

            #state = (random.randint(0, self.user_capacity), random.randint(0, self.user_capacity))
            #state = (0, 0)
            #while True:
            
            while action_taken < self.max_actions_per_episode:

                action = self.choose_action(state)
                next_state, reward = self.get_next_state_and_reward(state, action)
                self.update_Q(state, action, reward, next_state)
                total_reward += reward
                state = next_state
                action_taken += 1

            self.epsilon = max(0.1, self.epsilon * 0.995)
            rewards_per_episode.append(total_reward)
            print("Q-table:")
            print(self.Q)
            #cumulative_rewards += total_reward
            #print("Q-table:")
            #print(self.Q)
            #state = next_state
             #   if state[0] == 0:  # Stop when inventory reaches zero
             #       break
        print("Final Q-table:")
        print(self.Q)
        return rewards_per_episode
    def get_optimal_policy(self):
        # Derive the optimal policy from the trained Q-table
        optimal_policy = {}
        for state in self.Q.keys():
            #print(max(self.Q[state], key=self.Q[state].get))
            optimal_policy[state] = max(self.Q[state], key=self.Q[state].get)
        return optimal_policy
    #def get_optimal_value_function(self):
    #    # Get the optimal value function from the Q-table
    #    value_function = {state: max(actions) for state, actions in self.Q.items()}
    #    return pd.DataFrame.from_dict(value_function, orient='index', columns=['Value Function'])

# Example usage:
user_capacity = 2
poisson_lambda = 1.0
holding_cost = 1
stockout_cost = 10
gamma = 0.9
alpha = 0.001
epsilon = 0.1
episodes = 50
max_actions_per_episode = 100


ql = QLearningInventory(user_capacity, poisson_lambda, holding_cost, stockout_cost, gamma, 
                        alpha, epsilon, episodes, max_actions_per_episode)

ql.train()

optimal_policy = ql.get_optimal_policy()
#optimal_value_function = ql.get_optimal_value_function()

print("Optimal Policy:")
print(optimal_policy)

import matplotlib.pyplot as plt

rewards_per_episode = ql.train()

print("Rewards per Episode:")
print(rewards_per_episode)

print("Optimal Policy:")
print(optimal_policy)
# # Plot rewards per episode
plt.plot(range(1, episodes+1), rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode')
plt.show()
#print("\nOptimal Value Function:")
#print(optimal_value_function)

# Your dictionary
# q_values = {
#     (0, 0): {0: -50.43657596555572, 1: -65.177511969618315, 2: -55.84884176389377},
#     (0, 1): {0: -38.10601763127381, 1: -46.85623079297932},
#     (0, 2): {0: -32.14891398103115},
#     (1, 0): {0: -45.487078157023724, 1: -42.83954594645619},
#     (1, 1): {0: -32.96746473587354},
#     (2, 0): {0: -32.66353072084811}
# }

# # Loop to find and print the best action for each state

# state = (1, 0)
# best_action = max(q_values[state], key=q_values[state].get)

# # Output the best action
# print(best_action)
