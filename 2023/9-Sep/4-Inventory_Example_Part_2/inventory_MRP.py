from typing import Dict, Tuple
from scipy.stats import poisson
import numpy as np
import pandas as pd

class MarkovRewardProcess:
    def __init__(self):
        self.MRP_dict = {}

    def generate_Markov_Rew_Process_Dict(self, user_capacity: int, user_poisson_lambda: int,
                                          holding_cost: int, missedcostumer_cost: int):
        self.MRP_dict = {}  # Initialize the Markov Reward Process Dictionary

        for alpha in range(user_capacity + 1):
            for beta in range(user_capacity + 1 - alpha):
            
                state = (alpha, beta)
                init_inv = alpha + beta
                beta1 = user_capacity - init_inv

                base_reward = -alpha * holding_cost
            
                for demand in range(init_inv + 1):
                    if demand <= (init_inv - 1):
                        transition_prob = poisson.pmf(demand, user_poisson_lambda)
                                                
                        if state in self.MRP_dict:
                            self.MRP_dict[state][((init_inv - demand, beta1), base_reward)] = transition_prob
                        else:
                            self.MRP_dict[state] = {((init_inv - demand, beta1), base_reward): transition_prob}
                    else:
                        transition_prob = 1 - poisson.cdf(init_inv - 1, user_poisson_lambda)
                        transition_prob2 = 1 - poisson.cdf(init_inv, user_poisson_lambda)
                        reward = base_reward - missedcostumer_cost * ((user_poisson_lambda * transition_prob) -
                                                                init_inv * transition_prob2)
                        if state in self.MRP_dict:
                            self.MRP_dict[state][((0, beta1), reward)] = transition_prob
                        else:
                            self.MRP_dict[state] = {((0, beta1), reward): transition_prob}

    def calculate_expected_immediate_rewards(self):
        E_immediate_R = {}
        for from_state, value in self.MRP_dict.items():
            expected_reward = sum(reward[1] * prob for (reward, prob) in value.items())
            E_immediate_R[from_state] = expected_reward
        return E_immediate_R

    def create_transition_probability_matrix(self):
        states = list(self.MRP_dict.keys())
        num_states = len(states)
        trans_prob = np.zeros((num_states, num_states))
        df_trans_prob = pd.DataFrame(trans_prob, columns=states, index=states)

        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                for (new_state, reward) in self.MRP_dict.get(from_state, {}):
                    if new_state == to_state:
                        probability = self.MRP_dict[from_state].get((new_state, reward), 0.0)
                        df_trans_prob.iloc[i, j] = probability
        return df_trans_prob

    def calculate_state_value_function(self, trans_prob_mat, expected_immediate_rew, gamma):
        states = list(expected_immediate_rew.keys())
        R_exp = np.array(list(expected_immediate_rew.values()))
        val_func_vec = np.linalg.solve(np.eye(len(R_exp)) - gamma * trans_prob_mat, R_exp)
        MarkRevData = pd.DataFrame({'Expected Immediate Reward': R_exp, 'Value Function': val_func_vec}, index=states)
        return MarkRevData

mrp = MarkovRewardProcess()

# Generate the Markov Reward Process Dictionary
user_capacity = 2
user_poisson_lambda = 2
holding_cost = 1
missedcostumer_cost = 10

mrp.generate_Markov_Rew_Process_Dict(user_capacity, user_poisson_lambda, holding_cost, missedcostumer_cost)

E_immediate_R = mrp.calculate_expected_immediate_rewards()
trans_prob_mat = mrp.create_transition_probability_matrix()

gamma = 0.9  # Replace with your desired discount factor
MRP_Data = mrp.calculate_state_value_function(trans_prob_mat, E_immediate_R, gamma)

print(MRP_Data)