import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.specs.array_spec import BoundedArraySpec

from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.networks.q_network import QNetwork

from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.policy_saver import PolicySaver

from tf_agents.trajectories import time_step as ts

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()

'''
Position numbering conventions:

(00) Out of board

(03)--------------(02)--------------(01)
  |                 |                 |
  |   (11)--------(10)--------(09)    |
  |     |           |           |     |
  |     |   (19)--(18)--(17)    |     |
  |     |     |           |     |     |
(04)--(12)--(20)        (24)--(16)--(08)
  |     |     |           |     |     |
  |     |   (21)--(22)--(23)    |     |
  |     |           |           |     |
  |   (13)--------(14)--------(15)    |
  |                 |                 |
(05)--------------(06)--------------(07)

 1 = whites (agent)
-1 = blacks (opponent)
 0 = empty spot
'''


class NineMensMorris(PyEnvironment):

    # Initialize environment
    def __init__(self, agent_policy, discount=1.0):
        # Environment state:
        # Snapshot of game board
        # Flag indicating initial phase (men placement)
        self._state = {'board': np.zeros(24, dtype=np.int32), 'init_phase': np.int32(1)}
        # Step counter
        # Moves: total number of legal moves (excluding captures), to switch off initial phase
        # Draw: number of moves with both players having only 3 men; after 10, game ends (draw)
        self._step_counter = {'moves': np.int32(0), 'draw': np.int32(0)}
        # Adjacency matrix
        #                           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        self._adjacent = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                                   [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                                   [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                                   [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                                   [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
                                   [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                                   [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                                   [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
                                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 10
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 12
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 14
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 16
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # 17
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # 18
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # 19
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 20
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 21
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 22
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # 23
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]],  # 24
                                  dtype=np.int32)
        # Mills position matrix (listed as array indexes)
        self._mills = np.array([[0, 1, 2],
                                [2, 3, 4],
                                [4, 5, 6],
                                [6, 7, 0],
                                [8, 9, 10],
                                [10, 11, 12],
                                [12, 13, 14],
                                [14, 15, 8],
                                [16, 17, 18],
                                [18, 19, 20],
                                [20, 21, 22],
                                [22, 23, 16],
                                [1, 9, 17],
                                [3, 11, 19],
                                [5, 13, 21],
                                [7, 15, 23]],
                               dtype=np.int32)
        # Store the color with which the agent will play
        self._color = np.int32(1)
        # Store the desired discount rate (0 - 1)
        self._discount = np.float32(discount)
        # Initialize superclass
        super().__init__()
        # Random policy (used when the agent policy gives an illegal move)
        self._random_policy = RandomPyPolicy(self.time_step_spec(), self.action_spec())
        # Agent policy
        self._policy = agent_policy

    # Get actual game state
    def get_state(self):
        return self._state

    # Set actual game state
    def set_state(self, state):
        self._state = state

    # Get number of steps (state is already handled by get_state)
    def get_info(self):
        return self._step_counter

    # Return observation spec for the environment
    def observation_spec(self):
        # Observation spec:
        # Indexes 0 to 23 refer to the board
        # Index 24 tells the policy if the move must be a capture (1 = capture, 0 = no capture)
        # Index 25 tells if game is at initial phase (1 = initial, 0 = initial ended)
        return BoundedArraySpec(shape=(26,), dtype=np.int32, minimum=-1, maximum=1, name='observation')

    # Return action spec for the environment
    def action_spec(self):
        # Action spec:
        # A number between 0 and 624 representing the possible moves (mapped to "from" and "to" positions)
        return BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, maximum=624, name='action')

    # Reset the environment
    def _reset(self):
        # Create an empty board
        board_start = np.zeros(24, dtype=np.int32)
        # Agent has 50 % probability of starting the game
        start = np.random.rand()
        # Check the random value
        if start > 0.5:
            # Agent starts, board empty
            self.set_state({'board': board_start, 'init_phase': np.int32(1)})
            # Set step counters to zero
            self._step_counter = {'moves': np.int32(0), 'draw': np.int32(0)}
            # Return time step
            return ts.TimeStep(ts.StepType.FIRST, np.float32(0), np.float32(1), self.build_observation(0))
        else:
            # Randomly select a spot for the opponent
            board_start[np.random.randint(0, 24)] = -self._color
            self.set_state({'board': board_start, 'init_phase': np.int32(1)})
            # Set moves counter to 1 (opponent already played)
            self._step_counter = {'moves': np.int32(1), 'draw': np.int32(0)}
            # Return time step
            return ts.TimeStep(ts.StepType.MID, np.float32(0), np.float32(1), self.build_observation(0))

    # Build full observation
    def build_observation(self, capture):
        # Indexes 0 to 23 refer to the board
        # Index 24 tells the policy if the move must be a capture (1 = capture, 0 = no capture)
        # Index 25 tells if game is at initial phase (1 = initial, 0 = initial ended)
        return np.concatenate((self.get_state()['board'],
                               np.int32(capture),
                               self.get_state()['init_phase']), axis=None)

    # Count number of men still in game for a given color
    def men_count(self, color):
        n_men = np.int32(0)
        for i in range(0, 24):
            if self.get_state()['board'][i] == color:
                n_men += 1
        return n_men

    # Compute positions that form mills for a given color
    def mills_pos(self, color):
        mills_pos = np.zeros(24, dtype=np.int32)
        for i in range(0, 16):
            # Search for mills
            if np.array_equal(self.get_state()['board'][self._mills[i]], np.repeat(color, 3)):
                # Mark position
                mills_pos[self._mills[i]] = 1
        return mills_pos

    # Count number of mills for a given color
    def mills_count(self, color):
        mills_n = 0
        for i in range(0, 16):
            # Search for mills
            if np.array_equal(self.get_state()['board'][self._mills[i]], np.repeat(color, 3)):
                # Increment counter
                mills_n += 1
        return mills_n

    # Convert the numerical action (0 - 624) to new board configuration, given the color the agent is playing
    def action_to_board(self, action, color):
        # Copy actual board to new one
        new_board = np.copy(self.get_state()['board'])
        # Compute "from" (i) and "to" (j) positions
        i = action // 25
        j = action % 25
        # Translate "from"/"to" to new board
        if i == 0:
            # Action is from outside of the board (man placement)
            if j == 0:
                # Illegal move, board unchanged
                return self.get_state()['board']
            else:
                new_board[tuple(j - 1)] = color
        elif j == 0:
            # Action is to outside of the board (man capture)
            new_board[tuple(i - 1)] = 0
        else:
            # Action happens fully inside the board (man movement)
            new_board[tuple(i - 1)] = 0
            new_board[tuple(j - 1)] = color
        return new_board

    # Check if a mill was built by a given player (color), given an action
    def mill_built(self, action, color):
        # Compute new board from action
        new_board = self.action_to_board(action, color)
        # Count number of mills for the new board
        mills_new = 0
        for i in range(0, 16):
            # Search for mills
            if np.array_equal(new_board[self._mills[i]], np.repeat(color, 3)):
                # Increment counter
                mills_new += 1
        return mills_new > self.mills_count(color)

    # Compute allowed moves for a given color
    def allowed_moves(self, color):
        # Extract capture information from current observation
        capture = self.current_time_step().observation[24]
        # Start with empty matrix
        allowed_moves = np.zeros((25, 25), dtype=np.int32)
        # Check if current time step asks for a capture move
        if capture == 1:
            # Capturing means taking a man from the other color and moving to position 0 (outside of the board)
            # If the opponent has 3 or less men, choose any of them
            if self.men_count(-color) <= 3:
                for i in range(24):
                    if self.get_state()['board'][i] == -color:
                        allowed_moves[i + 1][0] = 1
            # If the opponent has more men, player is allowed to remove only the ones that don't form mills
            else:
                mills_opponent = self.mills_pos(-color)
                for i in range(24):
                    if self.get_state()['board'][i] == -color and mills_opponent[i] == 0:
                        allowed_moves[i + 1][0] = 1
        # No capture move: check if game is at initial phase
        elif self.get_state()['init_phase'] == 1:
            # At initial phase, player can place a man given that the chosen spot is empty
            for i in range(24):
                if self.get_state()['board'][i] == 0:
                    allowed_moves[0][i + 1] = 1
        # Initial phase ended
        else:
            # Check if player has 3 or less men
            if self.men_count(color) <= 3:
                # Player can "fly" to any empty spot
                for i in range(24):
                    for j in range(24):
                        if self.get_state()['board'][i] == color and self.get_state()['board'][j] == 0:
                            allowed_moves[i + 1][j + 1] = 1
            # Player has more than 3 men
            else:
                # Regular game: allowed to move to an adjacent free spot
                for j in range(24):
                    for i in range(24):
                        if self._adjacent[i + 1][j + 1] == 1:
                            if self.get_state()['board'][i] == color and self.get_state()['board'][j] == 0:
                                allowed_moves[i + 1][j + 1] = 1
        return allowed_moves

    # Check whether a certain move (action) is allowed
    def check_move(self, action, color):
        # Compute allowed moves matrix
        allowed_moves = self.allowed_moves(color)
        # Action number must match flattened matrix index
        return allowed_moves.flatten()[tuple(action)] == 1

    # Compute the outcome of the game for a given action
    def outcome(self, action, color):
        # Save the last state
        last_state = self.get_state()
        # Temporarily update the board
        self.update_board(action, color)
        for c in (-1, 1):
            # If a player has only 2 men (after initial phase), its opponent is the winner
            if self.men_count(c) <= 2 and self.get_state()['init_phase'] == 0:
                # Go back to previous state before returning
                self.set_state(last_state)
                return -c
            # If a player has no possible moves, its opponent is the winner
            elif np.count_nonzero(self.allowed_moves(c)) == 0:
                # Go back to previous state before returning
                self.set_state(last_state)
                return -c
        # If 10 or more legal moves occurred after both player have 3 or less men, the game draws
        if self.get_info()['draw'] >= 10:
            # Go back to previous state before returning
            self.set_state(last_state)
            return 0
        # No condition met: game continues
        else:
            # Go back to previous state before returning
            self.set_state(last_state)
            return None

    # Update the game board
    def update_board(self, action, color):
        self._state['board'] = self.action_to_board(action, color)

    # Flip colors on game board
    def flip_board(self):
        self._state['board'] = -self._state['board']

    # Compute the reward for a given action and player color
    def reward(self, action, color):
        reward_illegal = -1000
        reward_win = 10000
        reward_draw = 0
        reward_mill_built = 100
        reward_mill_opponent = -50
        reward_men = 10
        reward_men_opponent = 20
        # Check if move is legal
        if self.check_move(action, color):
            # Move is legal
            # Check the game outcome
            outcome = self.outcome(action, color)
            if outcome is not None:
                # Game ended
                if outcome == color:
                    # Victory
                    return np.float32(reward_win)
                elif outcome == -color:
                    # Loss
                    return np.float32(-reward_win)
                else:
                    # Draw
                    return np.float32(reward_draw)
            else:
                # Game not yet ended
                # Complete reward
                reward = self.mill_built(action, color) * reward_mill_built  # Mills built
                reward += self.mills_count(-color) * reward_mill_opponent  # Total number of opponent mills
                reward += self.men_count(color) * reward_men  # Total number of men
                if self.get_state()['init_phase'] == 0:  # Opponent men reward is computed only after initial phase
                    reward += (9 - self.men_count(-color)) * reward_men_opponent  # Total number of opponent's men
                return np.float32(reward)
        else:
            # Move is illegal
            return np.float32(reward_illegal)

    # Step the environment
    def _step(self, action):
        if self._current_time_step.is_last():
            return self.reset()
        else:
            # Extract capture information from current observation
            capture = self.current_time_step().observation[24]
            # Check if move is legal
            if self.check_move(action, self._color):
                # Move is legal
                # Increment moves counter if actual move is not a capture
                if capture == 0:
                    self._step_counter['moves'] += 1
                # Check number of moves to end initial phase (18 moves, 9 men placed by each color)
                if self._step_counter['moves'] == 18:
                    # Mark initial phase as concluded
                    self._state['init_phase'] = np.int32(0)
                # Increment draw counter if both player have 3 men, after initial phase
                if self.men_count(1) <= 3 and self.men_count(-1) <= 3 and self.get_state()['init_phase'] == 0:
                    self._step_counter['draw'] += 1
                # Compute reward
                reward = self.reward(action, self._color)
                # Compute step type
                if self.outcome(action, self._color) is not None:
                    # Game reached an outcome (victory or draw)
                    step_type = ts.StepType.LAST
                    # Update board
                    self.update_board(action, self._color)
                    # Return the final time step
                    return ts.TimeStep(step_type, reward, self._discount, self.build_observation(0))
                else:
                    # Game still going
                    step_type = ts.StepType.MID
                    # Check if a mill was built
                    if self.mill_built(action, self._color):
                        # Update board
                        self.update_board(action, self._color)
                        # Same color plays again, and next move must be a capture
                        return ts.TimeStep(step_type, reward, self._discount, self.build_observation(1))
                    else:
                        # No mill built: use the policy to create opponent play
                        # Update and flip board (other color plays)
                        self.update_board(action, self._color)
                        self.flip_board()
                        # Create a time step for the policy
                        policy_ts = ts.TimeStep(tf.convert_to_tensor([ts.StepType.MID]),
                                                tf.convert_to_tensor([reward]),
                                                tf.convert_to_tensor([self._discount]),
                                                tf.convert_to_tensor([self.build_observation(0)]))
                        random_ts = ts.TimeStep(ts.StepType.MID, reward, self._discount, self.build_observation(0))
                        # Retrieve the action from the policy
                        policy_action = self._policy.action(policy_ts).action
                        # Check if the policy action is a legal move
                        if not self.check_move(policy_action, self._color):
                            # Policy move is illegal
                            # Try a random move until it is legal
                            random_action = self._random_policy.action(random_ts).action
                            while not self.check_move(random_action, self._color):
                                random_action = self._random_policy.action(random_ts).action
                            # Replace the illegal policy action with the legal random action
                            policy_action = random_action
                        # Increment moves counter
                        self._step_counter['moves'] += 1
                        # Check number of moves to end initial phase (18 moves, 9 men placed by each color)
                        if self._step_counter['moves'] == 18:
                            # Mark initial phase as concluded
                            self._state['init_phase'] = np.int32(0)
                        # Increment draw counter if both player have 3 men, after initial phase
                        if self.men_count(1) <= 3 and self.men_count(-1) <= 3 and self.get_state()['init_phase'] == 0:
                            self._step_counter['draw'] += 1
                        # Compute reward for the policy
                        policy_reward = self.reward(policy_action, self._color)
                        # Compute step type
                        if self.outcome(policy_action, self._color) is not None:
                            # Game reached an outcome (victory or draw)
                            step_type = ts.StepType.LAST
                            # Update board
                            self.update_board(policy_action, self._color)
                            # Flip back board
                            self.flip_board()
                            # Return the final time step
                            return ts.TimeStep(step_type, -policy_reward, self._discount, self.build_observation(0))
                        else:
                            # Game still going
                            # Check if a mill was built
                            if self.mill_built(policy_action, self._color):
                                # Update board
                                self.update_board(policy_action, self._color)
                                # Policy must play again, and next move must be a capture
                                policy_ts = ts.TimeStep(tf.convert_to_tensor([ts.StepType.MID]),
                                                        tf.convert_to_tensor([policy_reward]),
                                                        tf.convert_to_tensor([self._discount]),
                                                        tf.convert_to_tensor([self.build_observation(1)]))
                                random_ts = ts.TimeStep(ts.StepType.MID, policy_reward, self._discount,
                                                        self.build_observation(1))
                                # Retrieve the action from the policy
                                policy_action = self._policy.action(policy_ts).action
                                # Check if the policy action is a legal move
                                if not self.check_move(policy_action, self._color):
                                    # Policy move is illegal
                                    # Try a random move until it is legal
                                    random_action = self._random_policy.action(random_ts).action
                                    while not self.check_move(random_action, self._color):
                                        random_action = self._random_policy.action(random_ts).action
                                    # Replace the illegal policy action with the legal random action
                                    policy_action = random_action
                                # Compute reward for the policy
                                policy_reward = self.reward(policy_action, self._color)
                                # Compute step type
                                if self.outcome(policy_action, self._color) is not None:
                                    # Game reached an outcome (victory or draw)
                                    step_type = ts.StepType.LAST
                                    # Update board
                                    self.update_board(policy_action, self._color)
                                    # Flip back board
                                    self.flip_board()
                                    # Return the final time step
                                    return ts.TimeStep(step_type, -policy_reward, self._discount,
                                                       self.build_observation(0))
                                else:
                                    # Game still going
                                    step_type = ts.StepType.MID
                                    # Update board
                                    self.update_board(policy_action, self._color)
                                    # Flip back board
                                    self.flip_board()
                                    # Return time step for the agent
                                    return ts.TimeStep(step_type, reward, self._discount, self.build_observation(0))
                            else:
                                # No mills built by the policy
                                # Game still going
                                step_type = ts.StepType.MID
                                # Update board
                                self.update_board(policy_action, self._color)
                                # Flip back board
                                self.flip_board()
                                # Return time step for the agent
                                return ts.TimeStep(step_type, reward, self._discount, self.build_observation(0))
            else:
                # Move is illegal
                # Compute step type
                if np.all(self.get_state()['board'] == 0):
                    # All positions are empty: start
                    step_type = ts.StepType.FIRST
                else:
                    # Illegal moves can't end the game
                    step_type = ts.StepType.MID
                # Illegal moves don't change capture spec, and same color plays again
                return ts.TimeStep(step_type, self.reward(action, self._color), self._discount,
                                   self.build_observation(capture))


# Parameters
DISCOUNT = 0.95  # Discount rate
BATCH_SIZE = 256  # Batch size (for replay buffer)
BUFFER_LENGTH = 131072  # Maximum number of steps in the buffer
STEPS_PER_ITER = 4096  # Steps collected per iteration (driver)
N_ITERATIONS = 1000  # Number of training iterations per session
EVAL_MAX_STEPS = 1000  # Maximum number of env steps during evaluation
COLLECT_RANDOM = True  # Use random policy to collect data

if __name__ == '__main__':
    # Create global step counter
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Create a dummy environment with no policy, just to extract the specs
    dummy_env = TFPyEnvironment(NineMensMorris(None, discount=DISCOUNT))

    # Create Q Network
    q_net = QNetwork(input_tensor_spec=dummy_env.observation_spec(),
                     action_spec=dummy_env.action_spec(),
                     fc_layer_params=(100, 600, 600, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 600, 600),
                     dropout_layer_params=(None, 0.1, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.1, None))

    # Create agent
    agent = DdqnAgent(time_step_spec=dummy_env.time_step_spec(),
                      action_spec=dummy_env.action_spec(),
                      q_network=q_net,
                      optimizer=Adam(learning_rate=1e-4),
                      td_errors_loss_fn=common.element_wise_squared_loss,
                      epsilon_greedy=0.1,
                      train_step_counter=global_step)
    # Initialize agent
    agent.initialize()
    # Wrap the training function in a TF graph
    agent.train = common.function(agent.train)

    # Create game environments: training and evaluation
    train_env = TFPyEnvironment(NineMensMorris(agent.policy, discount=DISCOUNT))
    eval_env = TFPyEnvironment(NineMensMorris(agent.policy, discount=DISCOUNT))

    # Random policy for data collection
    random_policy = RandomTFPolicy(time_step_spec=train_env.time_step_spec(),
                                   action_spec=train_env.action_spec())

    # Create replay buffer for data collection
    replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                          batch_size=train_env.batch_size,
                                          max_length=BUFFER_LENGTH)

    # Create driver for the agent
    driver = DynamicStepDriver(env=train_env,
                               policy=agent.collect_policy,
                               observers=[replay_buffer.add_batch],
                               num_steps=STEPS_PER_ITER)
    # Wrap the run function in a TF graph
    driver.run = common.function(driver.run)
    # Create driver for the random policy
    random_driver = DynamicStepDriver(env=train_env,
                                      policy=random_policy,
                                      observers=[replay_buffer.add_batch],
                                      num_steps=STEPS_PER_ITER)
    # Wrap the run function in a TF graph
    random_driver.run = common.function(random_driver.run)

    # Create a checkpointer
    checkpointer = common.Checkpointer(ckpt_dir=os.path.relpath('checkpoint'),
                                       max_to_keep=1,
                                       agent=agent,
                                       policy=agent.policy,
                                       replay_buffer=replay_buffer,
                                       global_step=global_step)
    checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()

    # Create a policy saver
    policy_saver = PolicySaver(agent.policy)

    # Main training loop
    time_step, policy_state = None, None
    for it in range(N_ITERATIONS):
        if COLLECT_RANDOM:
            print('Running random driver...')
            time_step, policy_state = random_driver.run(time_step, policy_state)
        print('Running agent driver...')
        time_step, policy_state = driver.run(time_step, policy_state)
        print('Training...')
        for train_it in range(BUFFER_LENGTH//BATCH_SIZE):
            experience, _ = replay_buffer.get_next(sample_batch_size=BATCH_SIZE, num_steps=2)
            agent.train(experience)
            if (train_it + 1) % 100 == 0:
                print('{0} training iterations'.format(train_it + 1))
        print('Saving...')
        # Save to checkpoint
        checkpointer.save(global_step)
        # Save policy
        policy_saver.save(os.path.relpath('policy'))
        # Show total reward of actual policy for 1 episode
        total_reward = 0.0
        eval_ts = eval_env.reset()
        num_steps = 0
        while (not eval_ts.is_last()) and num_steps < EVAL_MAX_STEPS:
            action_step = agent.policy.action(eval_ts)
            eval_ts = eval_env.step(action_step.action)
            total_reward += eval_ts.reward
            num_steps += 1
        print('Iteration = {0}: Steps taken: = {1} of {2}: Total reward = {3}'.format(it, num_steps,
                                                                                      EVAL_MAX_STEPS, total_reward))
