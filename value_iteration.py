import sys
from typing import Tuple

import numpy as np

if "../" not in sys.path:
    sys.path.append("../")


class ValueIteration:

    def __init__(self, prob_one: float, prob_three: float, prob_six: float, theta=0.0001, discount_factor=1.0):
        self.prob_one = prob_one
        self.prob_three = prob_three
        self.prob_six = prob_six
        self.theta = theta
        self.discount_factor = discount_factor


    def calculate_q_values(self, current_capital: int, value_function: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            current_capital: The gambler’s capital. Integer. (state)
            value_function: The vector that contains values at each state. (the recursive value function)
            rewards: The reward vector. (the immediate reward according to the gambler's problem definition)

        Returns:
            A vector containing the expected value of each action in THIS state.
            Its length equals to the number of actions.
        """
        # The maximum you can bet is limited by min(s, 100 - s).
        # Usually we skip bet=0 since it does nothing.
        max_bet = min(current_capital, 100 - current_capital)
        if max_bet == 0:
            # No valid bets => return an empty or single-value Q.
            # For terminal states s=0 or s=100, no real action anyway.
            return np.array([0.0])

        # Collect Q-values for each bet a in [1..max_bet]
        q_values = []
        for a in range(1, max_bet + 1):
            # Next states given each dice outcome:
            #  1 => capital goes down by 'a'
            #  3 => capital stays the same
            #  6 => capital goes up by 'a'
            next_s_if_1 = current_capital - a
            next_s_if_3 = current_capital
            next_s_if_6 = current_capital + a

            # Immediate rewards for those next states
            # (Often 0 for everything except reaching 100)
            r_if_1 = rewards[next_s_if_1]
            r_if_3 = rewards[next_s_if_3]
            r_if_6 = rewards[next_s_if_6]

            # Value of those next states
            v_if_1 = value_function[next_s_if_1]
            v_if_3 = value_function[next_s_if_3]
            v_if_6 = value_function[next_s_if_6]

            # Expected Q(s,a)
            expected_return = (self.prob_one * (r_if_1 + self.discount_factor * v_if_1)
                               + self.prob_three * (r_if_3 + self.discount_factor * v_if_3)
                               + self.prob_six * (r_if_6 + self.discount_factor * v_if_6))
            q_values.append(expected_return)

        return np.array(q_values)

    def value_iteration_for_gamblers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Value Iteration for the gambler’s problem until convergence.

        :return: (policy, V)
                 policy: np.ndarray of size 101, where policy[s] is the best bet at state s
                 V:      np.ndarray of size 101, the estimated value function at each state
        """

        # There are 101 states: 0..100
        # Typically, 0 and 100 are terminal states with V=0 (or V=1 for 100 if you prefer).
        # Here we'll just hold a 101-sized array for value function.
        V = np.zeros(101)
        # If your convention is to have reward=1 at state=100, you can either
        # incorporate that in V[100] or simply rely on immediate rewards array.
        # We'll rely on the immediate reward, so we keep V[100] = 0 initially.

        # Rewards: 1 if you reach 100, else 0
        rewards = np.zeros(101)
        rewards[100] = 1.0  # Winning state

        while True:
            delta = 0.0
            # Evaluate states 1..99 (0 and 100 are terminal)
            for s in range(1, 100):
                old_v = V[s]
                q_values = self.calculate_q_values(s, V, rewards)
                # Take the max over all feasible actions
                V[s] = np.max(q_values)
                delta = max(delta, abs(old_v - V[s]))

            # Check convergence
            if delta < self.theta:
                break

        # Once converged, extract the greedy policy
        policy = np.zeros(101, dtype=int)
        for s in range(1, 100):
            q_values = self.calculate_q_values(s, V, rewards)
            # The best action is the index of the max Q-value + 1
            # (since we enumerated actions from 1..max_bet)
            if len(q_values) > 1:
                best_a = np.argmax(q_values) + 1  # +1 because index 0 corresponds to bet=1
            else:
                best_a = 0
            policy[s] = best_a

        return policy, V
