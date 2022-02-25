from typing import Dict, List, Optional, Tuple

import haste_pytorch as haste
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Distribution
from torch.nn import functional as F


# Module to embed a single command
class CommandEmbedding(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.embedding = nn.Linear(input_size, output_size // 2)
    self.encoding = nn.Parameter(torch.rand(1, 1, output_size // 2))  # T x B x H

  def forward(self, command: Tensor) -> Tensor:
    return torch.cat([self.embedding(command), self.encoding.expand(command.size(0), command.size(1), -1)], dim=-1)


# Module to embed all (pre-specified) commands
class CommandEmbeddings(nn.Module):
  def __init__(self, goal_size: int, action_size: int, embedding_size: int):
    super().__init__()
    self.action_size = action_size
    self.goal_embedding = CommandEmbedding(goal_size, embedding_size)  # The goal is used as the "terminal indicator" in the meta-RL setting, so does not need to be dealt with separately if not mixing settings
    self.action_embedding = CommandEmbedding(action_size, embedding_size)
    self.reward_embedding = CommandEmbedding(1, embedding_size)
    self.desired_horizon_embedding = CommandEmbedding(1, embedding_size)
    self.desired_return_embedding = CommandEmbedding(1, embedding_size)

  def forward(self, commands: Dict[str, Tensor]) -> List[Tensor]:
    embeddings = [self.goal_embedding(commands['goal']), self.action_embedding(F.one_hot(commands['prev_action'], self.action_size).to(dtype=torch.float32)), self.desired_horizon_embedding(commands['desired_horizon'])]
    if 'reward' in commands: embeddings.append(self.reward_embedding(commands['reward']))
    if 'desired_return' in commands: embeddings.append(self.desired_return_embedding(commands['desired_return']))
    return embeddings


class Policy(nn.Module):
  def __init__(self, observation_size: int, action_size: int, goal_size: int, hidden_size: int):
    super().__init__()
    self.obs_fc1 = nn.Linear(observation_size, hidden_size)
    self.obs_fc2 = nn.Linear(hidden_size, hidden_size)
    self.command_embeddings = CommandEmbeddings(goal_size, action_size, hidden_size // 2)
    self.command_transformer = nn.TransformerEncoderLayer(hidden_size // 2, 1, hidden_size, dropout=0, activation='relu', norm_first=True)
    self.command_gating = nn.Linear(hidden_size // 2, hidden_size)
    self.rnn = haste.LayerNormLSTM(hidden_size, hidden_size)
    self.h_0, self.c_0 = nn.Parameter(torch.zeros(1, 1, hidden_size)), nn.Parameter(torch.zeros(1, 1, hidden_size))
    self.policy_fc = nn.Linear(hidden_size, action_size)

  def forward(self, observations: Tensor, commands: Dict[str, Tensor], hidden: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Distribution, Tuple[Tensor, Tensor]]:
    h1 = torch.tanh(self.obs_fc1(observations))
    embeddings = self.command_embeddings(commands)
    T, B, L = embeddings[0].size(0), embeddings[0].size(1), len(embeddings)
    embedding = torch.max(self.command_transformer(torch.stack(embeddings, dim=0).view(L, T * B, -1)), dim=0)[0].view(T, B, -1)  # Squeeze T x B before set equivariant op, then unsqueeze after set invariant op
    h2 = self.obs_fc2(h1) * torch.sigmoid(self.command_gating(embedding))
    if hidden is None: hidden = (self.h_0.expand(1, observations.size(1), -1), self.c_0.expand(1, observations.size(1), -1))
    h3, hidden = self.rnn(h2, hidden)
    logits = self.policy_fc(h3)
    return Categorical(logits=logits), hidden
