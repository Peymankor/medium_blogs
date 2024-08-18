import numpy as np
from scipy.stats import poisson
import pandas as pd

class MarkovDecisionProcess:
    def __init__(self, user_capacity, poisson_lambda, holding_cost, stockout_cost, gamma):
        # Initialize the MDP with given parameters
        self.user_capacity = user_capacity
        self.poisson_lambda = poisson_lambda
        self.holding_cost, self.stockout_cost = holding_cost, stockout_cost
        self.gamma = gamma
        self.full_MDP = self.create_full_MDP()  # Create the full MDP

    def create_full_MDP(self):
        # Create the full MDP dictionary
        MDP_dict = {}
        for alpha in range(self.user_capacity + 1):
            for beta in range(self.user_capacity + 1 - alpha):
                state, init_inv = (alpha, beta), alpha + beta 
                action = {}
                for order in range(self.user_capacity - init_inv + 1):
                    dict1 = {}
                    for i in range(init_inv + 1):
                        if i <= (init_inv - 1):
                            transition_prob = poisson.pmf(i, self.poisson_lambda)
                            dict1[((init_inv - i, order), -alpha * self.holding_cost)] = transition_prob
                        else:
                            transition_prob = 1 - poisson.cdf(init_inv - 1, self.poisson_lambda)
                            transition_prob2 = 1 - poisson.cdf(init_inv, self.poisson_lambda)
                            reward = -alpha * self.holding_cost - self.stockout_cost * (
                                (self.poisson_lambda * transition_prob) - init_inv * transition_prob2)
                            dict1[((0, order), reward)] = transition_prob
                    action[order] = dict1
                MDP_dict[state] = action
        return MDP_dict

    def policy_0_gen(self):
        # Generate an initial policy
        return {(alpha, beta): self.user_capacity - (alpha + beta) 
                for alpha in range(self.user_capacity + 1) 
                for beta in range(self.user_capacity + 1 - alpha)}

    def MRP_using_fixedPolicy(self, policy):
        # Create the MRP using a fixed policy
        return {state: self.full_MDP[state][action] 
                for state, action in policy.items()}
    
    def calculate_state_value_function(self, MRP_policy):
        # Calculate the expected immediate rewards from the MRP policy
        E_immediate_R = {}
        for from_state, value in MRP_policy.items():
            expected_reward = sum(reward[1] * prob for (reward, prob) in value.items())
            E_immediate_R[from_state] = expected_reward

        # Create the transition probability matrix
        states = list(MRP_policy.keys())
        trans_prob = np.zeros((len(states), len(states)))
        df_trans_prob = pd.DataFrame(trans_prob, columns=states, index=states)
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                for (new_state, reward) in MRP_policy.get(from_state, {}):
                    if new_state == to_state:
                        probability = MRP_policy[from_state].get((new_state, reward), 0.0)
                        df_trans_prob.iloc[i, j] = probability

        # Calculate the state value function
        R_exp = np.array(list(E_immediate_R.values()))
        val_func_vec = np.linalg.solve(np.eye(len(R_exp)) - self.gamma * df_trans_prob, R_exp)
        MarkRevData = pd.DataFrame({'Expected Immediate Reward': R_exp, 'Value Function': val_func_vec}, index=states)
        return MarkRevData

    def greedy_operation(self, MDP_full, state_val_policy, old_policy):
        # Perform the greedy operation to improve the policy
        new_policy = {}
        for state in old_policy.keys():
            max_q_value, best_action  = float('-inf'), None
            state_val_dict = state_val_policy.to_dict(orient="index")
            for action in MDP_full[state].keys():
                q_value = 0
                for (next_state, immediate_reward), probability in MDP_full[state][action].items():
                    q_value = q_value +  probability * (immediate_reward + self.gamma *
                        (state_val_dict[next_state]["Value Function"]))
                if q_value > max_q_value:
                    max_q_value, best_action = q_value, action
            new_policy[state] = best_action
        return new_policy

    def policy_iteration(self):
        # Perform policy iteration to find the optimal policy
        policy = self.policy_0_gen()
        while True:
            MRP_policy_p0 = self.MRP_using_fixedPolicy(policy)
            value_function = self.calculate_state_value_function(MRP_policy_p0)
            new_policy = self.greedy_operation(self.full_MDP, value_function, policy)
            if new_policy == policy:
                break
            policy = new_policy
        opt_policy, opt_value_func = new_policy, value_function
        return opt_policy, opt_value_func
    
#######

if __name__ == "__main__":
    user_capacity, poisson_lambda = 2, 1
    holding_cost, stockout_cost = 1.0, 10.0
    gamma = 0.9
    mdp = MarkovDecisionProcess(user_capacity, poisson_lambda, holding_cost, stockout_cost, gamma)
    opt_policy, opt_value_func = mdp.policy_iteration()
    print("Optimal Policy:")
    print(opt_policy)
    print("\nOptimal Value Function:")
    print(opt_value_func)