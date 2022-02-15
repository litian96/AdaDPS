import numpy as np

import torch
from torch.autograd import Variable

from .model_base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, params):
        super(Trainer, self).__init__(params)
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        # first and second moment estimate
        self.m = dict()
        self.v = dict()

        for p_name, p in self.model.named_parameters():
            self.m[p_name] = torch.zeros_like(p)
            self.v[p_name] = torch.zeros_like(p)

    def train(self):
        total_step = len(self.train_loader)
        for epoch in range(self.epochs):
            l_train = self.get_train_loss()
            l_test = self.get_test_loss()
            print('epoch {}, test loss {:.5f}, training loss {:.5f}'.format(epoch, l_test, l_train))


            for i, (images, labels) in enumerate(self.train_loader):
                images = images.reshape(-1, 784)
                images = Variable(images).to(self.device)
                reconstructed = self.model(images)
                l = self.loss(reconstructed, images)

                self.model.zero_grad()
                l.backward()

                for p_name, p in self.model.named_parameters():
                    self.m[p_name] = self.adam_beta_1 * self.m[p_name] + (1 - self.adam_beta_1) * p.grad
                    self.v[p_name] = self.adam_beta_2 * self.v[p_name] + (1 - self.adam_beta_2) * (p.grad ** 2)
                    m_hat = self.m[p_name] / (1 - np.power(self.adam_beta_1, total_step * epoch + i + 1))
                    v_hat = self.v[p_name] / (1 - np.power(self.adam_beta_2, total_step * epoch + i + 1))

                    p.grad = m_hat / (torch.sqrt(v_hat) + self.epsilon)

                self.optimizer.step()  # apply p.grad






