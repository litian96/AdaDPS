import random, os
import numpy as np
import argparse
import importlib

from generate_data import generate_mnist, generate_toy,generate_toy3


METHODS = ['sgd', 'adam', 'adagrad', 'rmsprop',
           'dp-sgd', 'dp-adam', 'dp-adagrad', 'dp-rmsprop']

def read_options():

    parser = argparse.ArgumentParser()
    parser.add_argument('--method',
                        help='training method',
                        type=str,
                        choices=METHODS,
                        default='sgd')
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=0.001)
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--iters',
                        type=int,
                        default=2250)
    parser.add_argument('--eval_every_iter',
                        type=int,
                        default=50)
    parser.add_argument('--seed',
                        help='numpy seed',
                        type=int,
                        default=3)
    parser.add_argument('--sigma',
                        help='sigma parameter (noise multiplier) of gaussian mechanism',
                        type=float,
                        default=0.7)
    parser.add_argument('--delta',
                        help='delta in the privacy parameters',
                        type=float,
                        default=1e-3)
    parser.add_argument('--clipping_bound',
                        help='max l2 norm of the gradient norm',
                        type=float,
                        default=1)
    parser.add_argument('--momentum',
                        help='whether to use momentum over (noisy) gradients',
                        type=int,
                        default=0)
    parser.add_argument('--momentum_parameter',
                        help='momentum parameter (close to 1)',
                        type=float,
                        default=0.9)
    parser.add_argument('--scale',
                        help='whether apply adadps',
                        type=int,
                        default=0)
    parser.add_argument('--epsilon',
                        help='the eps value in adaptive methods',
                        type=float,
                        default=1e-10)

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    opt = importlib.import_module('trainers.%s' % parsed['method'])
    trainer = getattr(opt, 'Trainer')
    print(parsed.items())

    np.random.seed(parsed['seed'])

    return parsed, trainer

def main():

    options, trainer = read_options()

    dataset = generate_toy()
    print('shape', dataset[0].shape)


    t = trainer(options, dataset)
    t.train()


if __name__ == "__main__":
    main()