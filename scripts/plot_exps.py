import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch


sns.set(style='white')


def stat_plot(x: np.array, y: np.array, mode: str=''):
  y_mean, y_std = np.transpose(np.mean(y, axis=0)), np.transpose(np.std(y, axis=0))
  palette = sns.color_palette('husl', y_mean.shape[0])
  for i in range(y_mean.shape[0]):
    ax = sns.lineplot(x=x, y=y_mean[i], color=palette[i])
    ax.fill_between(x, y_mean[i] + y_std[i], y_mean[i] - y_std[i], color=palette[i], alpha=0.1)
  if mode == 'offline':
    ax = sns.lineplot(x=x, y=np.full_like(x, 162), color='gray', linestyle='--')
    ax.fill_between(x, np.full_like(x, 162 + 195), np.full_like(x, 162 - 195), color='gray', alpha=0.1)

  plt.locator_params(axis='x', nbins=5)
  plt.locator_params(axis='y', nbins=6)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.xlabel('Step (x 1000)', fontsize=22)
  if mode in ['imitation', 'offline']:  # Make x-axis labels "invisible" for IL/offline RL (retains spacing, unlike removing the x ticks/label)
    ax.tick_params(axis='x', colors='white')
    ax.xaxis.label.set_color('white')
  plt.ylabel('Return', fontsize=22)
  plt.margins(x=0)
  plt.ylim(0, 500)
  plt.tight_layout()
  plt.savefig(f'results/{mode}/{mode}.png')
  plt.close()


train_start, step, test_interval = 10000, 200000, 10000

for mode in ['online', 'imitation', 'offline', 'goal', 'meta']:
  test_returns = []
  x = np.arange(100) if mode in ['imitation', 'offline'] else np.arange(train_start, step + 1, test_interval) / 1000
  for filename in os.listdir(f'results/{mode}'):
    if 'metrics.pth' in filename:
      test_returns.append(torch.load(f'results/{mode}/{filename}')['test_returns'])
      if mode in ['imitation', 'offline']: test_returns[-1] = 100 * test_returns[-1]
  y = np.array(test_returns)

  stat_plot(x, y, mode)
