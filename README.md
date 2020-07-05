# SEM5741 Neural Networks
Reinforcement learning implementations for the SEM5741 Neural Networks course at EESC-USP

# Prerequisites
All codes are written in Python and have TF-Agents (https://github.com/tensorflow/agents) and TensorFlow (https://github.com/tensorflow/tensorflow) as main dependencies. Install them via PIP packages, and configure CUDA support (https://www.tensorflow.org/install/gpu) if you have a NVIDIA GPU. These codes were successfully run using Python 3.7, TF-Agents 0.5 and TensorFlow 2.2 under Windows 10 x64. Refer to the fix by corypaik (https://github.com/tensorflow/agents/issues/219#issuecomment-551852360) to get ParallelPyEnvironment running under Windows (distributes data collection over CPU cores).

# Implementations
There are two main implementations here:
- Learning to play Nine Men's Morris via self-play using a DDQN agent. This implementation is incomplete and is still not able to learn. Future work may be carried on using a different agent (high probability of trying PPO).
- Learning to walk (Minitaur) with PPO. This code is fully functional and learns quite effectively. It is interesting to explore using different weights for the reward function and different network strucutres.
