from math import exp
from typing import Tuple

import gym
import numpy as np
import torch
from torch import Tensor


class Env():
  def __init__(self, mode: str):
    self.mode = mode
    self._env = gym.make('CartPole-v1')
    self._env.goal, self.goal_size = 0, 1  # Goal used as normal for GCRL setting, but as terminal indicator in meta-RL setting
    self.trial_ep, self.eps_per_trial = 0, 1  # Number of episodes per trial under meta-RL setting (note: not used here as meta-RL for generalisation is an established setting that makes more sense for CartPole)
    self.settings = [{'goal': 0, 'masscart': 1, 'total_mass': self._env.masspole + 1, 'length': 0.5, 'polemass_length': self._env.masspole * 0.5}]  # Default env params
    if self.mode == 'goal':
      goals = [-1, -0.5, 0, 0.5, 1]  # Evaluate a range of goals
      for goal in goals:
        setting = self.settings[0].copy()
        setting['goal'] = goal
        self.settings.append(setting)
      self.settings.pop(0)  # Remove default setting (already covered)
    elif self.mode == 'meta':
      lengths, masscarts = [0.5, 2.5, 4.5], [1, 2, 3]
      for length in lengths:
        for masscart in masscarts:
          setting = self.settings[0].copy()
          setting['masscart'], setting['total_mass'], setting['length'], setting['polemass_length'] = masscart, self._env.masspole + masscart, length, self._env.masspole * length
          self.settings.append(setting)
      self.settings.pop(0)  # Remove default setting (already covered)

  # Reset the environment and return first observation and goal
  def reset(self, setting: int=-1) -> Tuple[Tensor, Tensor]:
    if setting < 0:  # During training, randomise settings
      if self.mode == 'goal':
        self._env.goal = np.random.uniform(-1, 1)  # Sample a random goal location
      elif self.mode == 'meta':
        self._env.goal = 0  # Reset "terminal indicator"
        self._env.masscart, self._env.length = np.random.uniform(1, 4), np.random.uniform(1.5, 2.5)  # Randomise env params (on the inside of the tested bounds)
        self._env.total_mass, self._env.polemass_length = self._env.masspole + self._env.masscart, self._env.masspole * self._env.length  # Recalculate derived env params
    else:  # During testing, use a specified setting
      for k, v in self.settings[setting].items():
        setattr(self._env, k, v)
    observation = self._env.reset()
    return torch.as_tensor(observation, dtype=torch.float32).view(1, 1, -1), torch.tensor([[[self._env.goal]]], dtype=torch.float32)  # Return observation and goal with time and batch dims

  # Perform a step in the environment and return next observation, goal, reward and terminal signal
  def step(self, action: Tensor) -> Tuple[Tensor, float, bool, Tensor]:
    if self.mode == 'meta' and self._env.goal == 1:  # If in meta-RL setting and in-trial-episode has ended, reset environment (but not env params)
      self._env.goal = 0  # Reset "terminal indicator"
      observation, reward, terminal = self._env.reset(), 0, False
    else:
      observation, reward, terminal, _ = self._env.step(action[0, 0].numpy())  # Remove time and batch dims from action before passing into env
    if self.mode == 'goal':
      reward = exp(-abs(observation[0] - self._env.goal))  # Reward is inverse distance to goal
    elif self.mode == 'meta':
      if terminal:
        self.trial_ep += 1  # Increment in-trial-episode counter
        if self.trial_ep == self.eps_per_trial:  
          self.trial_ep, self._env.goal = 0, 1  # If trial is over then reset in-trial-episode counter and set "terminal indicator" (for the last time)
        else:
          terminal, self._env.goal = False, 1  # Overwrite terminal indicator and set "terminal indicator"
    return torch.as_tensor(observation, dtype=torch.float32).view(1, 1, -1), torch.tensor([[[self._env.goal]]], dtype=torch.float32), reward, terminal  # Return observation and goal with time and batch dims

  # Renders the environment state visually
  def render(self):
    self._env.render()

  @property
  def observation_size(self) -> gym.Space:
    return self._env.observation_space.shape[0]

  @property
  def action_size(self) -> gym.Space:
    return self._env.action_space.n
