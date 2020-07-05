from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imageio
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from tf_agents.environments import suite_pybullet
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment

from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.normal_projection_network import NormalProjectionNetwork
from tf_agents.networks.value_network import ValueNetwork

from tf_agents.agents.ppo.ppo_clip_agent import PPOClipAgent

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()

# Environment
env_name = "MinitaurBulletEnv-v0"


# Return a network built over the normal distribution (callable to be used in the actor network)
def normal_net(action_spec, i_m_o_f=0.1):
    return NormalProjectionNetwork(action_spec,
                                   mean_transform=None,
                                   state_dependent_std=True,
                                   init_means_output_factor=i_m_o_f,
                                   scale_distribution=True)


# Compute the average reward of a given policy for a given number of environment steps
def compute_total_reward(env: TFPyEnvironment, policy):
    total_reward = 0.0
    time_step = env.reset()
    while not time_step.is_last():
        policy_step = policy.action(time_step)
        time_step = env.step(policy_step.action)
        total_reward += time_step.reward
    return total_reward.numpy()[0]


# Create a callable for the environment, to enable parallel calls
def get_env():
    # The kwargs are the reward function weights
    return suite_pybullet.load(env_name, gym_kwargs={'distance_weight': 1.0,
                                                     'energy_weight': 0.001,
                                                     'shake_weight': 0.00,
                                                     'drift_weight': 0.05})


# Parameters
num_iter = 1000  # Number of iterations
collect_steps_per_iter = 8192  # Steps collected per iteration
replay_buffer_capacity = 16384  # Total number of steps stored by the replay buffer

# Info display intervals
eval_interval = 100

if __name__ == '__main__':

    # Load the environments
    eval_py_env = get_env()
    eval_env = TFPyEnvironment(eval_py_env)
    train_env = TFPyEnvironment(ParallelPyEnvironment([get_env] * 4, start_serially=False))

    # Create a global step
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Create the actor network (with the normal distribution)
    actor_net = ActorDistributionNetwork(input_tensor_spec=train_env.observation_spec(),
                                         output_tensor_spec=train_env.action_spec(),
                                         fc_layer_params=(128, 256, 512, 512, 256),
                                         continuous_projection_net=normal_net)

    # Create the value network
    value_net = ValueNetwork(input_tensor_spec=train_env.observation_spec(),
                             fc_layer_params=(256, 512, 512))

    # Create the PPO agent
    ppo_agent = PPOClipAgent(time_step_spec=train_env.time_step_spec(),
                             action_spec=train_env.action_spec(),
                             optimizer=Adam(learning_rate=5e-4),
                             actor_net=actor_net,
                             value_net=value_net,
                             importance_ratio_clipping=0.2,
                             discount_factor=0.95,
                             entropy_regularization=0.0,
                             num_epochs=16,
                             use_gae=True,
                             use_td_lambda_return=True,
                             log_prob_clipping=3,
                             gradient_clipping=0.5,
                             train_step_counter=global_step)
    # Initialize the agent
    ppo_agent.initialize()

    # Replay buffer (to store collected data)
    replay_buffer = TFUniformReplayBuffer(data_spec=ppo_agent.collect_data_spec,
                                          batch_size=train_env.batch_size,
                                          max_length=replay_buffer_capacity)

    # Create a checkpointer
    checkpointer = common.Checkpointer(ckpt_dir=os.path.relpath('ppo_checkpoint'),
                                       max_to_keep=1,
                                       agent=ppo_agent,
                                       policy=ppo_agent.policy,
                                       replay_buffer=replay_buffer,
                                       global_step=global_step)
    # Initialize the checkpointer
    checkpointer.initialize_or_restore()
    # Update the global step
    global_step = tf.compat.v1.train.get_global_step()

    # Create policy saver
    policy_saver = PolicySaver(ppo_agent.policy)

    # Create training driver
    train_driver = DynamicStepDriver(train_env,
                                     ppo_agent.collect_policy,
                                     observers=[replay_buffer.add_batch],
                                     num_steps=collect_steps_per_iter)
    # Wrap run function in TF graph
    train_driver.run = common.function(train_driver.run)
    print('Collecting initial data...')
    train_driver.run()

    # Reset the training step
    ppo_agent.train_step_counter.assign(0)

    # Evaluate the policy once before training
    print('Initial evaluation...')
    reward = compute_total_reward(eval_env, ppo_agent.policy)
    rewards = [reward]
    print('Initial total reward: {0}'.format(reward))

    # Number the videos (to view the evolution)
    video_num = 0

    # Training loop
    for it in range(num_iter):
        print('Running driver...')
        # Collect observations
        train_driver.run()
        # Update the agent's network
        print('Training (iteration {0})...'.format(it + 1))
        train_loss = ppo_agent.train(replay_buffer.gather_all())
        step = ppo_agent.train_step_counter.numpy()
        print('Step = {0}: Loss = {1}'.format(step, train_loss.loss))
        # Save to checkpoint
        checkpointer.save(global_step)
        if it % eval_interval == 0:
            reward = compute_total_reward(eval_env, ppo_agent.policy)
            print('Step = {0}: Average reward = {1}'.format(step, reward))
            rewards.append([reward])
            # Save policy
            policy_saver.save(os.path.relpath('ppo_policy'))
            # View a video of the robot
            video_filename = 'ppo_minitaur_{0}.mp4'.format(video_num)
            print('Creating video...')
            writer = imageio.get_writer(video_filename, fps=30)
            ts = eval_env.reset()
            writer.append_data(eval_py_env.render())
            while not ts.is_last():
                ts = eval_env.step(ppo_agent.policy.action(ts).action)
                writer.append_data(eval_py_env.render())
            writer.close()
            # Show the video
            os.startfile(video_filename)
            # Increment counter
            video_num += 1

    # View the average reward over training time
    steps = range(0, num_iter + 1, eval_interval)
    plt.plot(steps, rewards)
    plt.ylabel('Average reward')
    plt.xlabel('Step')
    plt.show()
