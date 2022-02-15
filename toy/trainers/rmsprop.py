import numpy as np

from .model_base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, params, data):
        super(Trainer, self).__init__(params, data)
        self.beta = 0.9
        self.v = np.zeros(len(self.data[0][0]))

    def train(self):
        for i in range(self.iters):
            idx = np.random.choice(range(len(self.data[0])), size=1)[0]
            g = self.get_gradients(idx)

            self.v = self.beta * self.v + (1 - self.beta) * np.square(g)
            g = np.divide(g, (np.sqrt(self.v)+self.epsilon))
            self.apply_gradients(g)
            if i % self.eval_every_iter == 0:
                print('iter', i, 'loss', self.get_loss())

        print(np.sqrt(self.v))
