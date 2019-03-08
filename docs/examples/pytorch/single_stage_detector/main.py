from argparse import ArgumentParser
from src.train import train

import torch
import numpy


def make_parser():
    parser = ArgumentParser(
        description="Train Single Shot MultiBox Detector on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco', required=True,
                        help='path to test and training data files')
    parser.add_argument('--epochs', '-e', type=int, default=65,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '--bs', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--eval-batch-size', '--ebs', type=int, default=32,
                        help='number of examples for each evaluation iteration')
    parser.add_argument('--seed', '-s', type=int, default=1,
                        help='manually set random seed for torch')
    parser.add_argument('--evaluation', nargs='*', type=int, default=[3, 21, 31, 37, 42, 48, 53, 59, 64],
                        help='epochs at which to evaluate')
    parser.add_argument('--multistep', nargs='*', type=int, default=[43, 54],
                        help='epochs at which to decay learning rate')

    # Hyperparameters
    parser.add_argument('--learning-rate', '--lr', type=float, default=2.6e-3,
                        help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='momentum argument for SGD optimizer')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0.0005,
                        help='weight decay value')
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--fp16', action='store_true')

    # Distributed
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')

    # Pipeline control
    parser.add_argument('--data_pipeline', type=str, default='dali',
                        choices=['dali', 'no_dali'],
                        help='data preprocessing pipline to use')

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    train(args)
