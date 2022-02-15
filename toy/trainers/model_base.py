import numpy as np


class BaseTrainer(object):
    def __init__(self, params, data):
        for key, val in params.items():
            setattr(self, key, val)
        self.data = data

        self.theta = np.random.normal(0, 0.1, len(self.data[0][0]))


    def get_loss(self):

        y_pred = np.dot(self.data[0], self.theta)
        loss = np.mean((self.data[1] - y_pred) ** 2 * 0.5)
        return loss


    def get_gradients(self, idx, batch_size):
        grads = []
        y_pred = np.dot(self.theta, self.data[0][idx:(idx+batch_size)].T)
        for i in range(batch_size):
            grad = self.data[0][idx+i] * (y_pred[i]-self.data[1][idx+i])
            grads.append(grad)
        return grads


    def apply_gradients(self, grads):
        self.theta = self.theta - self.lr * grads


