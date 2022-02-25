from typing import List, Optional, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


sns.set(style='white')


def lineplot(x: np.array, y: Union[List[float], List[List[float]]], filename: str, xlabel: str, ylabel: str, xlim: Optional[Tuple[float, float]]=None, ylim: Optional[Tuple[float, float]]=None, baseline_y: Optional[float]=None):
  if isinstance(y[0], list):
    y = np.transpose(np.array(y))
    palette = sns.color_palette('husl', y.shape[0])
    for i, y_i in enumerate(y):
      sns.lineplot(x=x, y=y_i, color=palette[i])
  else:
    sns.lineplot(x=x, y=y, color='coral')
  if baseline_y: sns.lineplot(x=x, y=np.full_like(x, baseline_y), color='gray', linestyle='--')

  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  if xlabel == '':
    plt.xticks([])  # Remove x-axis labels
  else:
    plt.xlabel(xlabel, fontsize=16)
  plt.ylabel(ylabel, fontsize=16)
  plt.xlim(*xlim) if xlim is not None else plt.margins(x=0)
  plt.ylim(*ylim) if ylim is not None else plt.margins(y=0)
  plt.tight_layout()
  plt.savefig(f'results/{filename}.png')
  plt.close()


def scatterplot(x: np.array, y: np.array, filename: str, xlabel: str, ylabel: str, xlim: Optional[Tuple[float, float]]=None, ylim: Optional[Tuple[float, float]]=None, calibration_line: bool=False):
  sns.scatterplot(x=x, y=y, color='coral')
  if calibration_line: sns.lineplot(x=np.arange(xlim[0], xlim[1]), y=np.arange(ylim[0], ylim[1]), color='gray', linestyle='--')

  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel(xlabel, fontsize=16)
  plt.ylabel(ylabel, fontsize=16)
  plt.xlim(*xlim) if xlim is not None else plt.margins(x=0)
  plt.ylim(*ylim) if ylim is not None else plt.margins(y=0)
  plt.tight_layout()
  plt.savefig(f'results/{filename}.png')
  plt.close()
