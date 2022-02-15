import numpy as np

from .model_base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, params, data):
        super(Trainer, self).__init__(params, data)
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        # first and second moment estimate
        self.m = np.zeros(len(self.data[0][0]))
        self.v = np.zeros(len(self.data[0][0]))

    def train(self):
        for i in range(self.iters):
            idx = np.random.choice(range(len(self.data[0])-self.batch_size), size=1)[0]
            grads = self.get_gradients(idx, self.batch_size)
            g = np.mean(np.array(grads), axis=0)

            self.m = self.adam_beta_1 * self.m + (1-self.adam_beta_1) * g
            self.v = self.adam_beta_2 * self.v + (1 - self.adam_beta_2) * np.square(g)
            m_hat = self.m / (1-np.power(self.adam_beta_1, i+1))
            v_hat = self.v / (1-np.power(self.adam_beta_2, i+1))
            g = np.divide(m_hat, (np.sqrt(v_hat)+self.epsilon))
            self.apply_gradients(g)
            if i % self.eval_every_iter == 0:
                print('iter', i, 'loss', self.get_loss())

        print(np.sqrt(self.v))
