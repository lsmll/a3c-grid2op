from __future__ import print_function

import argparse
import os

import grid2op

from agent import a3cAgent

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--hidden-size', type=int, default=256,
                    help='hidden layer size (default=256)')
parser.add_argument('--env-name', default='l2rpn_neurips_2020_track1_small',
                    help='environment to train on (default: l2rpn_neurips_2020_track1_small)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--for-test', default=False,
                    help='do testing with very small data instead of real data.')
parser.add_argument('--run-name', default=None,
                    help='tensorboard log dir name.')

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()
    env = grid2op.make(args.env_name, test=args.for_test)

    agent = a3cAgent(env.action_space, args)
    agent.train()
