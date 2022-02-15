import numpy as np
import argparse
import importlib


METHODS = ['sgd', 'adam', 'adagrad', 'rmsprop',
           'dp-sgd', 'dp-adam', 'dp-adagrad', 'dp-rmsprop']

def read_options():

    parser = argparse.ArgumentParser()
    parser.add_argument('--method',
                        help='training method',
                        type=str,
                        choices=METHODS,
                        default='sgd')
    parser.add_argument('--dataset',
                        help='name of the dataset: [mnist, emnist]',
                        type=str,
                        default='mnist')
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=0.1)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--num_microbatches',
                        help='how many microbatches in one mini-batch, only for dp methods',
                        type=int,
                        default=16)
    parser.add_argument('--epochs',
                        help='number of epochs',
                        type=int,
                        default=50)
    parser.add_argument('--iters',
                        help='number of mini-batch iterations',
                        type=int,
                        default=20000)
    parser.add_argument('--eval_every_epoch',
                        type=int,
                        default=1)
    parser.add_argument('--eval_every_iter',
                        type=int,
                        default=200)
    parser.add_argument('--seed',
                        help='numpy seed',
                        type=int,
                        default=0)
    parser.add_argument('--sigma',
                        help='sigma parameter of gaussian machenism (== noise_std / l2_bound)',
                        type=float,
                        default=1.0)
    parser.add_argument('--delta',
                        help='delta in the privacy parameters',
                        type=float,
                        default=1e-5)
    parser.add_argument('--clipping_bound',
                        help='max l2 norm of the gradient norm',
                        type=float,
                        default=1)
    parser.add_argument('--momentum',
                        help='whether to use momentum',
                        type=int,
                        default=0)
    parser.add_argument('--momentum_parameter',
                        help='momentum parameter (close to 1)',
                        type=float,
                        default=0.9)
    parser.add_argument('--use_public',
                        help='whether use public data to estimate E[g^2]',
                        type=int,
                        default=0)
    parser.add_argument('--public_bs',
                        help='batch size for the public dataset',
                        type=int,
                        default=10)
    parser.add_argument('--epsilon',
                        help='for preconditioner',
                        type=float,
                        default=1e-6)


    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    opt = importlib.import_module('trainers.%s' % parsed['method'])
    trainer = getattr(opt, 'Trainer')

    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)

    np.random.seed(parsed['seed'])

    return parsed, trainer

def main():

    options, trainer = read_options()

    t = trainer(options)
    t.train()


if __name__ == "__main__":
    main()