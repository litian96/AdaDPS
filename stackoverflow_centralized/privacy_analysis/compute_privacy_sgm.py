# compute privacy loss for subsampled gaussian mechanism
# code adapted from: https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/analysis

import math

import numpy as np

from privacy_analysis.rdp_accountant import compute_rdp
from privacy_analysis.rdp_accountant import get_privacy_spent

def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    """Compute and print results of DP-SGD analysis."""

    # compute_rdp requires that sigma be the ratio of the standard deviation of
    # the Gaussian noise to the l2-sensitivity of the function to which it is
    # added. Hence, sigma here corresponds to the `noise_multiplier` parameter
    # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
    rdp = compute_rdp(q, sigma, steps, orders)

    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

    print('DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
            ' over {} steps satisfies'.format(100 * q, sigma, steps), end=' ')
    print('differential privacy with eps = {:.3g} and delta = {}.'.format(eps, delta))
    print('The optimal RDP order is {}.'.format(opt_order))

    if opt_order == max(orders) or opt_order == min(orders):
        print('The privacy estimate is likely to be improved by expanding '
          'the set of orders.')

    return eps, opt_order


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
    """Compute epsilon based on the given hyperparameters.
    Args:
        n: Number of examples in the training data.
        batch_size: Batch size used in training.
        noise_multiplier: Noise multiplier used in training.
            (gaussian noise: var=sigma^2 C^2, sigma is the noise multiplier)
        epochs: Number of epochs in training.
        delta: Value of delta for which to compute epsilon.
    Returns:
        Value of epsilon corresponding to input hyperparameters.
    paper: https://arxiv.org/pdf/1908.10530.pdf Renyi Differential Privacy of the Sampled Gaussian Mechanism
    """
    q = batch_size / n  # q - the sampling ratio.
    orders = ([1.1, 1.2, 1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75] \
              + list(np.arange(5, 64, 0.5)) + [128, 256, 512])
    steps = int(math.ceil(epochs * n / batch_size))
    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)


