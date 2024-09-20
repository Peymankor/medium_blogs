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
    
    def test_policy(self, policy, episodes):
        """
        Test a given policy on the environment and calculate the total reward.

        Args:
            policy (dict): A dictionary mapping states to actions.
            episodes (int): The number of episodes to simulate.

        Returns:
            float: The total reward accumulated over all episodes.
        """
        total_reward = 0
        alpha_0 = random.randint(0, self.user_capacity)
        beta_0 = random.randint(0, self.user_capacity - alpha_0)
        state = (alpha_0, beta_0)  # Initialize the state
        
        for _ in range(episodes):

            action = policy.get(state, 0)
            next_state, reward = self.simulate_transition_and_reward(state, action)
            total_reward += reward
            state = next_state

        return total_reward

# Example usage:
## Define the parameters
user_capacity = 10
poisson_lambda = 4
holding_cost = 8
stockout_cost = 10
gamma = 0.9
alpha = 0.1
epsilon = 0.1
episodes = 1000
max_actions_per_episode = 1000


## Define the Class
ql = QLearningInventory(user_capacity, poisson_lambda, holding_cost, stockout_cost, gamma, 
                        alpha, epsilon, episodes, max_actions_per_episode)

## Train Agent
ql.train()

## Get the Optimal Policy
optimal_policy = ql.get_optimal_policy()



# Visualize the qlearning policy

states = list(optimal_policy.keys())

## Number of states
n_states = len(states)

## Create positions for bars with a small offset
x = np.arange(n_states)  # Positions for the states on the x-axis

## Create a grouped bar chart
fig, ax = plt.subplots(figsize=(14, 6))

## Plot optimal policy bars
ax.bar(x, [optimal_policy[state] for state in states], 
       label=' Q-Learning Policy Order', color='purple')

## Plot simple policy bars next to the optimal policy bars

## Set labels and titles
ax.set_xlabel('State (alpha: on-hand inventory, beta: on-order inventory)')
ax.set_ylabel('Number of Order')
ax.set_xticks(x)  # Center the ticks between the bars
ax.set_xticklabels(states, rotation=90)
ax.set_title('Number of Orders for each State from Q-Learning Policy')

## Add legend
ax.legend()

## Display plot
plt.tight_layout()

## Save the plot after displaying it
plt.savefig('./Fig/qlearningpolicy.png', dpi=300)



# Create a simple policy
def order_up_to_policy(state, user_capacity, target_level):
    alpha, beta = state
    max_possible_order = user_capacity - (alpha + beta)
    desired_order = max(0, target_level - (alpha + beta))
    return min(max_possible_order, desired_order)


target_level = 10
simple_policy = {state: order_up_to_policy(state, user_capacity, target_level) for state in optimal_policy.keys()}



# Visualize the simple policy and Q -learning

# Create positions for bars with a small offset
bar_width = 0.35  # Width of each bar
x = np.arange(n_states)  # Positions for the states on the x-axis

# Create a grouped bar chart
fig, ax = plt.subplots(figsize=(14, 6))

# Plot optimal policy bars
ax.bar(x, [optimal_policy[state] for state in states], width=bar_width, color='purple', label=' Q-Learning Policy Order')

# Plot simple policy bars next to the optimal policy bars
ax.bar(x + bar_width, [simple_policy[state] for state in states], width=bar_width, color='orange', label='Simple Policy Order')

# Set labels and titles
ax.set_xlabel('State (alpha: on-hand inventory, beta: on-order inventory)')
ax.set_ylabel('Number of Order')
ax.set_xticks(x + bar_width / 2)  # Center the ticks between the bars
ax.set_xticklabels(states, rotation=90)
ax.set_title('Comparison of Orders Between Q-Learning Policy and Simple Policy')

# Add legend
ax.legend()

# Display plot
plt.tight_layout()

# Save the plot in high quality
plt.savefig('./Fig/plot_comp.png', dpi=300)


# Test the optimal policy vs a simple policy
episodes_val = 10000
total_reward_qlearning = ql.test_policy(optimal_policy, episodes_val)
print("Total Reward with Q-Learning Policy", total_reward_qlearning)

target_level = 10
simple_policy = {state: order_up_to_policy(state, user_capacity, target_level) for state in optimal_policy.keys()}

# Test the simple policy
total_reward_simp = ql.test_policy(simple_policy, episodes_val)
print("Total Reward with Simple Policy:", total_reward_simp)

fig, ax = plt.subplots(figsize=(14, 6))

# Create a bar plot with narrower bars
labels = ['Q-Learning Policy', 'Simple Policy']
rewards = [-total_reward_qlearning, -total_reward_simp]

plt.bar(labels, rewards, width=0.4)  # Set the width to 0.4
plt.xlabel('Policy')
plt.ylabel('Total Cost of Running Inventory')
plt.title('Comparison of Total Costs Following Two Policies')

plt.savefig('./Fig/plot_comp_rewards.png', dpi=300)

plt.show()