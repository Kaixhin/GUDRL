from copy import deepcopy
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from agent import Agent
from environment import Env
from memory import Memory


def test(agent: Agent, memory: Memory, mode: str, num_episodes: int, init_command: Optional[Tensor]=None) -> float:
  env = Env(mode)
  returns = []

  for setting_id in range(len(env.settings)):
    setting_returns = []

    for episode in tqdm(range(1, num_episodes + 1), leave=False):
      (observation, goal), terminal, total_reward = env.reset(setting=setting_id), False, 0
      command, hidden = agent.get_initial_command(goal, memory, mode) if init_command is None else deepcopy(init_command), None
      while not terminal:
        with torch.inference_mode(): policy, hidden = agent.observe(observation, command, hidden)
        action = policy.sample()
        next_observation, goal, reward, terminal = env.step(action)
        agent.update_command(observation, goal, action, reward, terminal, command, mode)
        total_reward += reward
        observation = next_observation
      setting_returns.append(total_reward)

    returns.append(np.mean(setting_returns))

  return returns


def calibration(agent: Agent, memory: Memory, mode: str, num_episodes: int) -> Tuple[List[float], List[float]]:
  achieved_returns, desired_returns = [], [100, 200, 300, 400, 500]
  command = agent.get_initial_command(torch.zeros(1, 1, 1), memory, mode)

  for desired_return in desired_returns:
    command['desired_horizon'], command['desired_return'] = torch.tensor([[[desired_return]]], dtype=torch.float32), torch.tensor([[[desired_return]]], dtype=torch.float32)  # Desired horizon is same as desired return for CartPole
    achieved_returns.append(test(agent, memory, mode, num_episodes, command)[0])

  return achieved_returns, desired_returns
