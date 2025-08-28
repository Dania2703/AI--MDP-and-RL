# Reinforcement Learning Projects – Q-Learning & Value Iteration

This repository contains two classic reinforcement learning algorithms applied to simple problems.

## Overview
- **Q-Learning:** An off-policy reinforcement learning algorithm that learns the optimal action-value function by interacting with an environment. The implementation trains an agent to take actions in a discrete state space (like environments from Gymnasium) using an epsilon-greedy policy for exploration.
- **Value Iteration (Gambler’s Problem):** A dynamic programming algorithm that iteratively improves the value function until convergence. The implementation solves the Gambler’s Problem, where a player bets portions of their capital and the algorithm finds the optimal policy to maximize the probability of reaching the goal.

## Purpose
These projects demonstrate how reinforcement learning can be implemented from scratch with NumPy. They show:
- How an agent can learn by trial and error using Q-Learning.
- How optimal strategies can be computed directly using Value Iteration.
- The differences between learning from experience (Q-Learning) and planning with a known model (Value Iteration).

## Applications
- Teaching and practicing reinforcement learning fundamentals.
- Understanding how agents balance exploration and exploitation.
- Applying dynamic programming to Markov Decision Processes (MDPs).
- Serving as a foundation for more complex RL projects.

## Requirements
- Python 3.9+
- NumPy
- Gymnasium (for the Q-Learning environment)

## License
MIT (free to use and modify).
