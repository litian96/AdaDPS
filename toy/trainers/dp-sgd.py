import numpy as np

from .model_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import *

class Trainer(BaseTrainer):

    def __init__(self, params, data):
        super(Trainer, self).__init__(params, data)
        self.n, self.d = len(self.data[0]), len(self.data[0][0])
        self.delta = 1.0 / len(self.data)

    def train(self):

        A = np.ones(self.d)
        A[int(self.d / 10):] = 0.01

        for j in range(5):
            print('trial ', j)
            np.random.seed(123+j)
            self.theta = np.random.normal(0, 0.1, len(self.data[0][0]))

            for i in range(self.iters):
                idx = np.random.choice(range(len(self.data[0])-self.batch_size), size=1)[0]
                grads = self.get_gradients(idx, self.batch_size)
                final_grad = np.zeros_like(grads[0])
                for grad in grads:
                    if self.scale:
                        grad = grad / (A + 1e-3)
                    # clip the gradient
                    grad = grad / np.maximum(1, np.linalg.norm(grad) / self.clipping_bound)
                    final_grad += grad

                final_grad = final_grad + np.random.normal(0, self.sigma * self.clipping_bound, size=len(grads[0]))
                final_grad /= self.batch_size

                self.apply_gradients(final_grad)

                if i % self.eval_every_iter == 0:
                    print('iter', i, 'loss', self.get_loss())


            compute_dp_sgd_privacy(len(self.data[0]), self.batch_size, \
                                     self.sigma, self.iters/len(self.data[0]), self.delta)

