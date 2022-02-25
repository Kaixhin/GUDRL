from argparse import ArgumentParser


def get_argparser() -> ArgumentParser:
  parser = ArgumentParser(description='Generalised UDRL')
  parser.add_argument('--seed', type=int, default=0, help='Random seed')
  parser.add_argument('--mode', type=str, choices=['online', 'imitation', 'offline', 'goal', 'meta'], default='online', help='RL mode')
  parser.add_argument('--steps', type=int, default=200000, help='Total environment steps')
  parser.add_argument('--offline-iter', type=int, default=30000, help='Imitation/offline training iterations')
  parser.add_argument('--mem-size', type=int, default=5, help='Episodic memory size')
  parser.add_argument('--hidden-size', type=int, default=128, help='Policy hidden size')
  parser.add_argument('--batch-size', type=int, default=32, help='Minibatch size')
  parser.add_argument('--seq-len', type=int, default=10, help='Minibatch sequence length')
  parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
  parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
  parser.add_argument('--train-start', type=int, default=10000, help='Step at which training starts')
  parser.add_argument('--train-interval', type=int, default=20, help='Training interval')
  parser.add_argument('--test-interval', type=int, default=10000, help='Testing interval')
  parser.add_argument('--test-episodes', type=int, default=10, help='Testing episodes')
  parser.add_argument('--log-interval', type=int, default=10000, help='Logging interval')
  parser.add_argument('--render', action='store_true', help='Render environment')
  parser.add_argument('--save-mem', action='store_true', help='Save episodic memory')
  return parser
