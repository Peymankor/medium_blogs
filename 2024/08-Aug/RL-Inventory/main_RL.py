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
                    
                    Q[state][action] = np.random.uniform(0, 1)  # Start with small random values instead of 0 
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

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.user_capacity - 
                                    (state[0] + state[1]) + 1)  # Explore
        else:

            best_action= max(self.Q[state], key=self.Q[state].get)
            
            return best_action

    def update_Q(self, state, action, reward, next_state):
        # Update the Q-table based on the state, action, reward, and next state
        best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)
       
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train(self):
        # Train the Q-learning algorithm over the specified number of episodes
        rewards_per_episode = []
        #cumulative_rewards = 0

        for episode in range(self.episodes):
            
            #state = (0, 0)
            alpha_0 =  random.randint(0, self.user_capacity)
            beta_0 = random.randint(0, self.user_capacity - alpha_0)
            state = (alpha_0, beta_0)  # Initialize the state at the start of each episode
            total_reward = 0  # Initialize the total reward for the current episode
            action_taken = 0
            
            while action_taken < self.max_actions_per_episode:

                action = self.choose_action(state)
                next_state, reward = self.get_next_state_and_reward(state, action)
                #print("State:", next_state)
                self.update_Q(state, action, reward, next_state)
                total_reward += reward
                state = next_state
                action_taken += 1

            self.epsilon = max(0.01, self.epsilon * 0.995)
            rewards_per_episode.append(total_reward)
          
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

# Example usage:
user_capacity = 10
poisson_lambda = 10
holding_cost = 1
stockout_cost = 5
gamma = 0.99
alpha = 0.05
epsilon = 0.01
episodes = 1000
max_actions_per_episode = 100


ql = QLearningInventory(user_capacity, poisson_lambda, holding_cost, stockout_cost, gamma, 
                        alpha, epsilon, episodes, max_actions_per_episode)

ql.train()

optimal_policy = ql.get_optimal_policy()
#optimal_value_function = ql.get_optimal_value_function()

#print("Optimal Policy:")
#print(optimal_policy)

import numpy as np


def simulate_environment(episodes, user_capacity, poisson_lambda, 
                         optimal_policy, holding_cost, stockout_cost,
                         policy_to_act):
    """
    Test the optimal policy on the new environment and calculate the total reward.

    Args:
        episodes (int): The number of episodes to simulate.
        user_capacity (int): The maximum user capacity.
        poisson_lambda (float): The lambda parameter for the Poisson distribution.
        optimal_policy (dict): A dictionary mapping states to actions.
        holding_cost (float): The cost per unit for holding inventory.
        stockout_cost (float): The cost per unit for stockout.

    Returns:
        float: The total reward accumulated over all episodes.
    """
    total_reward = 0

    alpha_0 =  random.randint(0, user_capacity)
    beta_0 = random.randint(0, user_capacity - alpha_0)
    state = (alpha_0, beta_0)  # Initialize the state at the start of each episode
    
    for _ in range(episodes):

        action = policy_to_act.get(state, 0)
        alpha, beta = state
        demand = np.random.poisson(poisson_lambda)
        new_alpha = max(0, alpha + beta - demand)
        next_state = (new_alpha, action)

        #action, demand, next_state = new_environment(user_capacity, 
        #                                            poisson_lambda, 
        #                                            state, action)
        #alpha, beta = state
        #action = optimal_policy.get(state, 0)
        
        init_inv = alpha + beta  # Initial inventory (alpha + items on order)
        #new_alpha = max(0, init_inv - demand)
        
        # Calculate the reward
        holding_cost_value = -new_alpha * holding_cost
        stockout_cost_value = -(demand - init_inv) * stockout_cost if demand > init_inv else 0
        
        reward = holding_cost_value + stockout_cost_value
        total_reward += reward
        
        state = next_state

    return total_reward

def constrained_order_up_to_policy(state, user_capacity, target_level):
    alpha, beta = state
    # Determine the maximum quantity you can order without exceeding capacity
    max_possible_order = user_capacity - (alpha + beta)
    # Calculate the desired order quantity to reach the target level
    desired_order = max(0, target_level - (alpha + beta))
    # The action is the minimum of what is possible and what is desired
    return min(max_possible_order, desired_order)

# Generate the simple policy with the capacity constraint
target_level = 5
simple_policy = {state: constrained_order_up_to_policy(state, user_capacity, target_level) for state in optimal_policy.keys()}

#simple_policy = {state: user_capacity-state[0]-state[1] for state in optimal_policy.keys()}
#simple_policy = {state: 1 for state in optimal_policy.keys()}
#print("Simple Policy:")
#print(simple_policy)

#print("Optimal Policy:")
#print(optimal_policy)

episodes_val = 10000

total_reward_opt = simulate_environment(episodes_val, user_capacity, 
                                    poisson_lambda, optimal_policy, 
                                    holding_cost, stockout_cost,
                                    policy_to_act=optimal_policy)  

print("Total Reward on New Environment: Optimum", total_reward_opt)

total_reward_simp = simulate_environment(episodes_val, user_capacity, 
                                    poisson_lambda, optimal_policy, 
                                    holding_cost, stockout_cost,
                                    policy_to_act=simple_policy)  

print("Total Reward on New Environment Simple Polciy:", total_reward_simp)

#print("Total Reward on New Environment:", total_reward)
#import matplotlib.pyplot as plt

#rewards_per_episode = ql.train()

#print("Rewards per Episode:")
#print(rewards_per_episode)

#print("Optimal Policy:")
#print(optimal_policy)
# # Plot rewards per episode
#plt.plot(range(1, episodes+1), rewards_per_episode)
#plt.xlabel('Episode')
#plt.ylabel('Reward')
#plt.title('Rewards per Episode')
#plt.show()
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

import matplotlib.pyplot as plt
import numpy as np  # Make sure to import NumPy

# Sample data (assuming simple_policy and optimal_policy are dictionaries)
# Replace this with your actual data
#simple_policy = {'State 1': 2, 'State 2': 3, 'State 3': 1}
#optimal_policy = {'State 1': 3, 'State 2': 4, 'State 3': 2}

# Get the keys and values of the simple policy
simple_keys = list(simple_policy.keys())
simple_values = np.array(list(simple_policy.values()))  # Convert to NumPy array

# Get the keys and values of the optimal policy
optimal_keys = list(optimal_policy.keys())
optimal_values = np.array(list(optimal_policy.values()))  # Convert to NumPy array

# Ensure both policies have the same states
assert simple_keys == optimal_keys, "The keys (states) of both policies must match."

# Number of bars
n = len(simple_keys)

# Create a range for the bars
bar_width = 0.35
index = np.arange(n)

# Create the bar chart
plt.bar(index, simple_values, bar_width, label='Simple Policy')
plt.bar(index + bar_width, optimal_values, bar_width, label='Optimal Policy')

# Rotate x-axis labels for better visibility
plt.xticks(index + bar_width / 2, simple_keys, rotation=45)

# Add legend
plt.legend()
plt.xlabel('State')
plt.ylabel('Action')
plt.title('Comparison of Simple Policy and Optimal Policy')
plt.legend()

# Show the plot
plt.show()
