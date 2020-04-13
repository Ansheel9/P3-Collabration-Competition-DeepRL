---
title: 'Report'
disqus: Ansheel Banthia
---

Project 3 - Collabration & Competition
===

Train a pair of ML-Agents to play Tennis :tennis:

Udacity Deep Reinforcement Learning Nanodegree

## Table of Contents

[TOC]

## Summary

The project is based on Unity Environment. The agent is trained to play tennis in multi-agent environment!

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
 - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
 - This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

Learning Algorithm
---

For artificial intelligence (AI) to reach its full potential, AI systems need to interact safely and efficiently with humans, as well as other agents. There are already environments where this happens on a daily basis, such as the stock market. And there are future applications that will rely on productive agent-human interactions, such as self-driving cars and other autonomous vehicles.

One step along this path is to train AI agents to interact with other agents in both cooperative and competitive settings. Reinforcement learning (RL) is a subfield of AI that's shown promise. However, thus far, much of RL's success has been in single agent domains, where building models that predict the behavior of other actors is unnecessary. As a result, traditional RL approaches (such as Q-Learning) are not well-suited for the complexity that accompanies environments where multiple agents are continuously interacting and evolving their policies.

#### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

The original DDPG algorithm from which I extended to create the MADDPG version, is outlined in [this paper](https://arxiv.org/pdf/1509.02971.pdf), Continuous Control with Deep Reinforcement Learning, by researchers at Google Deepmind. In this paper, the authors present "a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces." They highlight that DDPG can be viewed as an extension of Deep Q-learning to continuous tasks.

For the DDPG foundation, I used vanilla, single-agent DDPG as a template. Then, to make this algorithm suitable for the multiple competitive agents in the Tennis environment, I implemented components discussed in [this paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf), Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, by Lowe and Wu, along with other researchers from OpenAI, UC Berkeley, and McGill University.

#### Actor-Critic Method

Actor-critic methods leverage the strengths of both policy-based and value-based methods. Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent. Meanwhile, employing a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. Actor-critic methods combine these two approaches in order to accelerate the learning process. Actor-critic agents are also more stable than value-based agents, while requiring fewer training samples than policy-based agents.


There are also a few techniques which contributed significantly towards stabilizing the training:
- Fixed Q-target: 2 different networks are combined in order to keep the method off-policy. A target network is updated at every iteration while an evalution network at the end of updating phase.
- Experience Replay: In order to decouple sequential states of each episode, Replay buffer <S,a,R,S'> is created. At each iteration a random batch is pulled from buffer.
- Soft Updates: In DQN, the target networks are updated by copying all the weights from the local networks after a certain number of epochs. However, in DDPG, the target networks are updated using soft updates where during each update step, 0.01% of the local network weights are mixed with the target networks weights.

I used Linear Neural Network architecture. Also, in my experience, I have found Batch normalization to have always improved training and hence, I added Batch normalization layer in both actor and critic.

Parameters used in MADDPG algorithm:
- Maximum steps per episode: 1000
- Batch Size: 256
- Buffer size: 1e4
- Gamma: 0.99
- Learning rate (Actor): 1e-4
- Learning rate (Critic): 1e-3
- tau: 1e-3
- Initial Noise Weighting Factor: 0.5
- Noise Decay Rate: 1.0
- Weight Decay: 0

Result
---

Here is a plot that shows the score over the episodes, as the agent is trained.

![](plot.png)

The environment was solved in 3293 episodes, achieving an average score of 0.50 over the past 100 episodes.

Ideas of Future Work
---

- Other algorithms like Proximal Policy Approximation (PPO) that have been discussed in the course could potentially lead to good results as well.
- Using Prioritized Experience Replay which has generally shown to have been quite useful. It is expected that it'll lead to an improved performance here too.
- Apply this agent to different Cooperative and Competitive environments, such as the Soccer playing agents.
