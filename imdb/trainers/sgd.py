import numpy as np

import torch

from .model_base import BaseTrainer
from opacus import PrivacyEngine

class Trainer(BaseTrainer):

    def __init__(self, params):
        super(Trainer, self).__init__(params)
        self.mean = dict()
        self.preconditioner = dict()
        for p_name, p in self.model.named_parameters():
            self.mean[p_name] = torch.zeros_like(p)
            self.preconditioner[p_name] = torch.zeros_like(p)


    def train(self):
        total_step = len(self.train_loader)

        if self.scale_by == 'tfidf':
            freq = self.get_tf_idf_value()
            multiplier = freq * 10
            multiplier[:3] = 0.01
        elif self.scale_by == 'freq':
            freq = self.get_bow_frequency()
            multiplier = np.minimum(self.multiplier_cap, np.divide(max(freq), freq + 1e-20)) / self.division
        elif self.scale_by == 'importance':  # this is not used, only for development
            importance = np.load('diagnostics/adam_weight.npy')  # adam weight is obtained under the BOW features
            importance = np.mean(abs(importance), axis=0)
            multiplier = (importance / min(importance)) / self.division

        multiplier = torch.FloatTensor(multiplier).cuda()

        for epoch in range(self.epochs):

            if epoch % self.eval_every_epoch == 0:
                train_accu, train_l = self.get_train_accuracy_and_loss()
                accu = self.get_test_accuracy()
                print('epoch {}, test accuracy {:.5f}, training accuracy {:.5f}, training loss {:.5f}' \
                      .format(epoch, accu, train_accu, train_l))
                self.model.train()  # turn on training


            if self.use_public:
                self.estimate_preconditioner()

            for i, (x, labels) in enumerate(self.train_loader):

                x = x.to(self.device)
                labels = labels.to(self.device)
                predicted = self.model(x)
                l = self.loss(predicted, labels)
                self.model.zero_grad()
                l.backward()

                if self.scale:
                    for p_name, p in self.model.named_parameters():
                        if p_name == 'weight':
                            p.grad = p.grad * multiplier

                if self.use_public:
                    for p_name, p in self.model.named_parameters():
                        p.grad = p.grad / torch.sqrt(self.preconditioner[p_name]+self.epsilon)

                # if self.momentum:
                #     for p_name, p in self.model.named_parameters():
                #         self.mean[p_name] = self.momentum_parameter * self.mean[p_name] + (1 - self.momentum_parameter) * p.grad
                #         p.grad = self.mean[p_name]

                self.optimizer.step()  # apply p.grad
