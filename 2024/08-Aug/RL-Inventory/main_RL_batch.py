import numpy as np
import random
import matplotlib.pyplot as plt

class QLearningInventory:
    def __init__(self, user_capacity, poisson_lambda, holding_cost, stockout_cost, 
                 gamma, alpha, epsilon, episodes, max_actions_per_episode):
        
        # Initialize parameters
        self.user_capacity = user_capacity
        self.poisson_lambda = poisson_lambda
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_actions_per_episode = max_actions_per_episode
        self.batch = []  # Initialize the batch to store experiences

    def initialize_Q(self):
        # Initialize the Q-table as a dictionary
        Q = {}
        for alpha in range(self.user_capacity + 1):
            for beta in range(self.user_capacity + 1 - alpha):
                state = (alpha, beta)
                Q[state] = {}
                max_action = self.user_capacity - (alpha + beta)
                for action in range(max_action + 1):
                    Q[state][action] = np.random.uniform(0, 1)  # Small random values
        return Q

    def simulate_transition_and_reward(self, state, action):

        alpha, beta = state
        init_inv = alpha + beta
        demand = np.random.poisson(self.poisson_lambda)
        
        new_alpha = max(0, init_inv - demand)
        holding_cost = -new_alpha * self.holding_cost
        stockout_cost = 0
        
        if demand > init_inv:
            stockout_cost = -(demand - init_inv) * self.stockout_cost
        
        reward = holding_cost + stockout_cost
        next_state = (new_alpha, action)
        
        return next_state, reward

    def choose_action(self, state):

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.user_capacity - (state[0] + state[1]) + 1)
        else:
            return max(self.Q[state], key=self.Q[state].get)

    def update_Q(self, batch):
        # Batch update of the Q-table
        for state, action, reward, next_state in batch:
            best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)
            td_target = reward + self.gamma * self.Q[next_state][best_next_action]
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error

    def train(self):

        self.Q = self.initialize_Q()  # Reinitialize Q-table for each training run

        for episode in range(self.episodes):
            alpha_0 = random.randint(0, self.user_capacity)
            beta_0 = random.randint(0, self.user_capacity - alpha_0)
            state = (alpha_0, beta_0)
            #total_reward = 0
            self.batch = []  # Reset the batch at the start of each episode
            action_taken = 0
            while action_taken < self.max_actions_per_episode:
                action = self.choose_action(state)
                next_state, reward = self.simulate_transition_and_reward(state, action)
                self.batch.append((state, action, reward, next_state))  # Collect experience
                state = next_state
                action_taken += 1
            
            self.update_Q(self.batch)  # Update Q-table using the batch
            

    def get_optimal_policy(self):
        optimal_policy = {}
        for state in self.Q.keys():
            optimal_policy[state] = max(self.Q[state], key=self.Q[state].get)
        return optimal_policy

# Example usage:
user_capacity = 10
poisson_lambda = 5
holding_cost = 2
stockout_cost = 5
gamma = 0.9
alpha = 0.1
epsilon = 0.1
episodes = 1000
max_actions_per_episode = 1000


# Define the Class
ql = QLearningInventory(user_capacity, poisson_lambda, holding_cost, stockout_cost, gamma, 
                        alpha, epsilon, episodes, max_actions_per_episode)

# Train Agent
ql.train()

# Get the Optimal Policy
optimal_policy = ql.get_optimal_policy()

# Print the Policy
print("Optimal Policy:")

for state, action in optimal_policy.items():
    alpha, beta = state
    print(f"State: alpha={alpha}, beta={beta}, Optimum Order: {action}")


states = list(optimal_policy.keys())
opt_actions = list(optimal_policy.values())

plt.figure(figsize=(14, 6))
plt.bar(range(len(states)), opt_actions)
plt.xlabel('State (alpha: on-hand inventory, beta:on-order inventory )')
plt.ylabel('Optimum Order')
plt.bar(range(len(states)), opt_actions, color='purple')  # Change the color of the bars to purple
plt.xticks(range(len(states)), states, rotation=90)  # Rotate x-axis labels by 90 degrees
plt.title('Optimal Action for Each State')
#plt.grid(True)  # Add grid lines to the plot
plt.xticks(range(len(states)), states, rotation=90)  # Rotate x-axis labels by 90 degrees
plt.title('Optimal Order for Each State')
#plt.show()

plt.savefig('/home/peyman/Documents/Creator/medium_blogs/2024/08-Aug/RL-Inventory/optimal_policy_plot.png', dpi=300, bbox_inches='tight')

#visualize_action(optimal_policy)
#import numpy as np


#def simulate_environment(episodes, user_capacity, poisson_lambda, 
#                         optimal_policy, holding_cost, stockout_cost,
#                         policy_to_act):
#    """
#    Test the optimal policy on the new environment and calculate the total reward.

#    Args:
#        episodes (int): The number of episodes to simulate.
#        user_capacity (int): The maximum user capacity.
#        poisson_lambda (float): The lambda parameter for the Poisson distribution.
#        optimal_policy (dict): A dictionary mapping states to actions.
#        holding_cost (float): The cost per unit for holding inventory.
#        stockout_cost (float): The cost per unit for stockout.

#    Returns:
#        float: The total reward accumulated over all episodes.
#    """
#    total_reward = 0

#    alpha_0 =  random.randint(0, user_capacity)
#    beta_0 = random.randint(0, user_capacity - alpha_0)
#    state = (alpha_0, beta_0)  # Initialize the state at the start of each episode
    
#    for _ in range(episodes):

#       action = policy_to_act.get(state, 0)
#        alpha, beta = state
#        demand = np.random.poisson(poisson_lambda)
#        new_alpha = max(0, alpha + beta - demand)
#        next_state = (new_alpha, action)

        #action, demand, next_state = new_environment(user_capacity, 
        #                                            poisson_lambda, 
        #                                            state, action)
        #alpha, beta = state
        #action = optimal_policy.get(state, 0)
        
#        init_inv = alpha + beta  # Initial inventory (alpha + items on order)
        #new_alpha = max(0, init_inv - demand)
        
        # Calculate the reward
#        holding_cost_value = -new_alpha * holding_cost
#        stockout_cost_value = -(demand - init_inv) * stockout_cost if demand > init_inv else 0
        
#        reward = holding_cost_value + stockout_cost_value
#        total_reward += reward
        
#        state = next_state

#    return total_reward

#episodes_val = 10000

#total_reward_opt = simulate_environment(episodes_val, user_capacity, 
#                                    poisson_lambda, optimal_policy, 
#                                    holding_cost, stockout_cost,
#                                    policy_to_act=optimal_policy)  

#print("Total Reward on New Environment: Optimum", total_reward_opt)

#target_level = user_capacity

#def constrained_order_up_to_policy(state, user_capacity, target_level):
#    alpha, beta = state
    # Determine the maximum quantity you can order without exceeding capacity
#    max_possible_order = user_capacity - (alpha + beta)
    # Calculate the desired order quantity to reach the target level
#    desired_order = max(0, target_level - (alpha + beta))
    # The action is the minimum of what is possible and what is desired
#    return min(max_possible_order, desired_order)

#simple_policy = {state: constrained_order_up_to_policy(state, user_capacity, target_level) for state in optimal_policy.keys()}


#total_reward_simp = simulate_environment(episodes_val, user_capacity, 
#                                    poisson_lambda, optimal_policy, 
#                                    holding_cost, stockout_cost,
#                                    policy_to_act=simple_policy)  


#rint("Total Reward on New Environment Simple Polciy:", total_reward_simp)

#import matplotlib.pyplot as plt

#def visualize_action(optimal_policy):
#    states = list(optimal_policy.keys())
#    actions = list(optimal_policy.values())

#    plt.bar(range(len(states)), actions)
#    plt.xlabel('State')
#    plt.ylabel('Action')
#    plt.xticks(range(len(states)), states, rotation=90)  # Rotate x-axis labels by 90 degrees
#    plt.title('Optimal Action for Each State')
#    plt.show()

#visualize_action(optimal_policy)