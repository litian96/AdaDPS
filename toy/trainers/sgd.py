import numpy as np

from .model_base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, params, data):
        super(Trainer, self).__init__(params, data)
        self.n, self.d = len(self.data[0]), len(self.data[0][0])

    def train(self):

        A = np.ones(self.d)
        A[int(self.d / 10):] = 0.01

        for j in range(5):
            print('trial ', j)
            np.random.seed(123+j)

            for i in range(self.iters):
                idx = np.random.choice(range(len(self.data[0])-self.batch_size), size=1)[0]
                grads = self.get_gradients(idx, self.batch_size)
                g = np.mean(np.array(grads), axis=0)
                if self.scale:
                    g = g / (A+1e-3)
                self.apply_gradients(g)
                if i % self.eval_every_iter == 0:
                    print('iter', i, 'loss', self.get_loss())
