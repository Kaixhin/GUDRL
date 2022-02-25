from typing import Dict, List, Tuple

import numpy as np
from sortedcontainers import SortedList
import torch
from torch import Tensor


class Memory():
  def __init__(self, size: int, observation_size: int, action_size: int, goal_size: int):
    self.current_episode = []
    self.episodes, self.max_episodes = SortedList(key=lambda x: -x[0]), size
    self.blank_observation, self.blank_action, self.blank_goal = torch.zeros(observation_size), torch.tensor(0, dtype=torch.int64), torch.zeros(goal_size)

  # Appends a transition to the current episode buffer; stores whole episode upon termination
  def update(self, observation: Tensor, goal: Tensor, action: Tensor, reward: float, terminal: bool):
    self.current_episode.append((observation[0, 0], goal[0, 0], action[0, 0], reward))  # Remove time and batch dims from observation, goal and action before storing
    if terminal:
      episode = {'observations': torch.stack([self.blank_observation] + [transition[0] for transition in self.current_episode]), 'goals': torch.stack([self.blank_goal] + [transition[1] for transition in self.current_episode]), 'actions': torch.stack([self.blank_action] + [transition[2] for transition in self.current_episode]), 'rewards': torch.tensor([0] + [transition[3] for transition in self.current_episode], dtype=torch.float32)}  # Stack transitions and prepend with blank transition
      episode['returns'] = torch.flip(torch.flip(episode['rewards'], dims=(0, )).cumsum(dim=0), dims=(0, ))  # Calculate and store returns
      episode['length'] = episode['rewards'].size(0) - 1  # Calculate and store episode length

      self.episodes.add((episode['returns'][0].item(), episode))  # Add (return, trajectory) to episode priority queue
      if len(self.episodes) > self.max_episodes: self.episodes.pop()  # Keep fixed size episode priority queue
      self.current_episode = []  # Clear current episode buffer

  # Samples a batch of subtrajectories (observations, commands, actions) from a given episode
  def _sample(self, idx: int, batch_size: int, seq_len: int, mode: str) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
    episode = self.episodes[idx][1]
    seq_len = min(seq_len, episode['length'] - 1)  # Cap sequence length if necessary
    idxs = np.random.randint(0, episode['length'] - seq_len, (batch_size, )) + 1  # Select from any valid transition (episode prepended with blank transition for previous action)
    idxs = np.repeat(idxs, seq_len) + np.tile(np.arange(seq_len), batch_size)  # Create unrolled sequence of indices for extracting subtrajectories

    observations = episode['observations'][idxs].view(seq_len, batch_size, -1)
    actions = episode['actions'][idxs].view(seq_len, batch_size)
    commands = {'goal': episode['goals'][idxs].view(seq_len, batch_size, -1), 'prev_action': episode['actions'][idxs - 1].view(seq_len, batch_size), 'desired_horizon': torch.as_tensor(episode['length'] - (idxs - 1)).view(seq_len, batch_size, 1).to(dtype=torch.float32)}
    if mode != 'imitation':
      commands['reward'], commands['desired_return'] = episode['rewards'][idxs].view(seq_len, batch_size, 1), episode['returns'][idxs].view(seq_len, batch_size, 1)

    return observations, commands, actions

  # Samples a batch of subtrajectories (observations, commands, actions) from the episodes with the max returns
  def sample(self, batch_size: int, seq_len: int, mode: str) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
    idx = np.random.randint(len(self.episodes))
    return self._sample(idx, batch_size, seq_len, mode)

  # Returns the return and horizon of the episode with the max return
  def get_max_return_horizon(self) -> List[float]:
    return [self.episodes[0][0], self.episodes[0][1]['length']] if len(self.episodes) > 0 else [0, 0]

  # Returns the average return
  def get_avg_return(self) -> float:
    return np.mean([episode[0] for episode in self.episodes])

  # Saves a list of episodes
  def save(self, filename: str):
    torch.save(list(iter(self.episodes)), f'results/{filename}.pth')

  # Loads episodes (based on training mode); overwrites size
  def load(self, filename: str, mode: str):
    episodes = torch.load(filename)
    if mode == 'imitation':
      episodes = episodes[:5]  # Select top 5 episodes (avg. reward 500 +/- 0)
    elif mode == 'offline':
      episodes = episodes[-1000:]  # Select bottom 1000 episodes (avg. reward 162 +/- 195)
    self.episodes.clear()
    self.episodes.update(episodes)
