# Train a pair of ML-Agents to play Tennis :tennis:

---

Udacity Deep Reinforcement Learning Nanodegree

Project 3 - Collabration & Competition

---

## Introduction

The project is based on Unity Environment. The Agent is trained to play tennis in a multi-agent environment!

| Before Training | After Training |
| :---: | :---: |
| ![](https://github.com/Ansheel9/P3-Collabration-Competition-DeepRL/blob/master/Images/BT.gif) | ![](https://github.com/Ansheel9/P3-Collabration-Competition-DeepRL/blob/master/Images/AT.gif) |

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
 - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
 - This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

---

## Getting Started

### Environment
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - MacOS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the 'p3_collab-compet/' folder, and unzip (or decompress) the file.

3. Install conda environment with  <code> conda env create -f environment.yml </code>

### Dependencies
 
 - Create (and activate) a new environment with Python.\
 <code> conda create --name drlnd </code>\
 <code> activate drlnd </code>
 
 - Install Pytorch by following the instructions in [this link.](https://pytorch.org/get-started/locally/)
 
 - Other dependencies required for this program are listed in the <code> requirements.txt </code> file so that you can install them using the following command:\
 <code> pip install requirements.txt </code>

---

## Instruction

To train the agent, start jupyter notebook, open <code> Tennis.ipynb </code> and execute! For more information, please check instructions inside the notebook.

---

## Result

Plot showing the average score over 100 episode.

![](https://github.com/Ansheel9/P3-Collabration-Competition-DeepRL/blob/master/Images/plot.PNG)
