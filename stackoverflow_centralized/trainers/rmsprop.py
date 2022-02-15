import numpy as np

import torch

from .model_base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, params):
        super(Trainer, self).__init__(params)
        self.second = dict()
        self.beta = 0.9

        for p_name, p in self.model.named_parameters():
            self.second[p_name] = torch.zeros_like(p)

    def train(self):

        for epoch in range(self.epochs):

            if epoch % self.eval_every_epoch == 0:
                l = self.get_train_loss()
                accu = self.get_test_accuracy()
                print('epoch {}, test accuracy {:.5f}, training loss {:.5f}'.format(epoch, accu, l))
                self.model.train()  # turn on training

            for i, (x, labels) in enumerate(self.train_loader):

                x = x.to(self.device)
                labels = labels.to(self.device)

                predicted = self.model(x)
                l = self.loss(predicted, labels)
                self.optimizer.zero_grad()
                l.backward()

                for p_name, p in self.model.named_parameters():
                    self.second[p_name] = self.beta * self.second[p_name] + (1-self.beta) * (p.grad ** 2)
                    p.grad = p.grad / (torch.sqrt(self.second[p_name]) + self.epsilon)

                self.optimizer.step()  # apply p.grad


