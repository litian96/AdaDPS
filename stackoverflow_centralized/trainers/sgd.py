import numpy as np
import copy

import torch
from torch.nn.utils import clip_grad_norm_

from .model_base import BaseTrainer

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


        freq = np.load('diagnostics/frequency.npy')
        freq = np.where(freq < 1e-20, np.average(freq[9000:10000]), freq)
        multiplier = np.minimum(10000000, np.divide(max(freq), freq + 1e-20))
        multiplier = torch.FloatTensor(multiplier).cuda()

        for epoch in range(self.epochs):

            if epoch % self.eval_every_epoch == 0:
                #l = self.get_train_loss()
                accu = self.get_test_accuracy()
                #print('epoch {}, test accuracy {:.5f}, training loss {:.5f}'.format(epoch, accu, l))
                print('epoch {}, test accuracy {:.5f}'.format(epoch, accu))
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
                        p.grad = p.grad / torch.sqrt(self.preconditioner[p_name]+1e-10)

                if self.momentum:
                    for p_name, p in self.model.named_parameters():
                        self.mean[p_name] = self.momentum_parameter * self.mean[p_name] + (1 - self.momentum_parameter) * p.grad
                        p.grad = copy.deepcopy(self.mean[p_name])

                self.optimizer.step()  # apply p.grad



