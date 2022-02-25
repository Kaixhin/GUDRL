from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import RMSprop

from memory import Memory
from policy import Policy


class Agent():
  def __init__(self, observation_size: int, action_size: int, goal_size: int, hidden_size: int, learning_rate: float, weight_decay: float):
    self.policy = Policy(observation_size, action_size, goal_size, hidden_size)
    self.optimiser = RMSprop(self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay, alpha=0.7)

  # Sets training/evaluation mode
  def set_mode(self, training: bool):
    self.policy.train(training)

  # Constructs an initial command for a new episode Dict[str, T x B x C]
  def get_initial_command(self, goal: Tensor, memory: Memory, mode: str) -> Dict[str, Tensor]:
    desired_return, desired_horizon = memory.get_max_return_horizon()
    command = {'goal': goal, 'prev_action': torch.zeros(1, 1, dtype=torch.int64), 'desired_horizon': torch.tensor([[[desired_horizon]]], dtype=torch.float32)}
    if mode != 'imitation':
      command['reward'], command['desired_return'] = torch.zeros(1, 1, 1), torch.tensor([[[desired_return]]], dtype=torch.float32)
    return command

  # Observes the current state and produces a policy and updated internal/hidden state
  def observe(self, observations: Tensor, commands: Dict[str, Tensor], hidden: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Distribution, Tuple[Tensor, Tensor]]:
    return self.policy(observations, commands, hidden)

  # Updates command (inplace)
  def update_command(self, observation: Tensor, goal: Tensor, action: Tensor, reward: float, terminal: bool, command: Tensor, mode: str) -> Tensor:
    command['goal'], command['prev_action'], command['desired_horizon'] = goal, action, command['desired_horizon'] - 1  # Update goal, previous action, and subtract one timestep from desired horizon
    if mode != 'imitation':
      command['reward'], command['desired_return'] = torch.tensor([[[reward]]], dtype=torch.float32), command['desired_return'] - reward  # Update reward and subtract reward from desired return

  # Trains on past observations using supervised learning
  def train(self, memory: Memory, batch_size: int, seq_len: int, mode: str) -> float:
    observations, commands, actions = memory.sample(batch_size, seq_len, mode)
    policy, _ = self.policy(observations, commands, None)

    self.optimiser.zero_grad(set_to_none=True)
    loss = -policy.log_prob(actions).mean()
    loss.backward()
    self.optimiser.step()
    return loss.item()
