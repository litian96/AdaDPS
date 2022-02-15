import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from .model_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import *
import copy


class Trainer(BaseTrainer):
    def __init__(self, params):
        super(Trainer, self).__init__(params)
        self.delta = 1.0 / 25000
        self.beta = 0.9
        # second moment estimate
        self.second = dict()
        for p_name, p in self.model.named_parameters():
            self.second[p_name] = torch.zeros_like(p)

    def train(self):
        total_step = len(self.train_loader)
        for epoch in range(self.epochs):

            if epoch % self.eval_every_epoch == 0:
                train_accu, train_l = self.get_train_accuracy_and_loss()
                accu = self.get_test_accuracy()
                print('epoch {}, test accuracy {:.5f}, training accuracy {:.5f}, training loss {:.5f}' \
                      .format(epoch, accu, train_accu, train_l))
                print('epoch {}, test accuracy {:.5f}'.format(epoch, accu))
                # compute eps based on RDP
                compute_dp_sgd_privacy(total_step * self.batch_size, self.batch_size, self.sigma, epoch, self.delta)
                print()
                self.model.train()  # back to training mode

            for i, (x, labels) in enumerate(self.train_loader):

                x = x.to(self.device)
                labels = labels.to(self.device)
                predicted = self.model(x)
                l = self.loss_flat(predicted, labels)

                if len(l) % self.num_microbatches == 0:  # handle corner cases
                    losses = torch.mean(l.reshape(self.num_microbatches, -1), dim=1)
                else:
                    continue

                saved_var = dict()
                for p_name, p in self.model.named_parameters():
                    saved_var[p_name] = torch.zeros_like(p)

                for j in losses:  # for every micro-batch in the mini-batch
                    self.model.zero_grad()
                    j.backward(retain_graph=True)
                    clip_grad_norm_(self.model.parameters(), self.clipping_bound)
                    for p_name, p in self.model.named_parameters():
                        new_grad = p.grad
                        saved_var[p_name].add_(new_grad)

                for p_name, p in self.model.named_parameters():
                    if self.device.type == 'cuda':
                        noise = torch.cuda.FloatTensor(p.grad.shape).normal_(0, self.sigma * self.clipping_bound)
                    else:
                        noise = torch.FloatTensor(p.grad.shape).normal_(0, self.sigma * self.clipping_bound)
                    saved_var[p_name].add_(noise)
                    p.grad = saved_var[p_name] / self.num_microbatches

                for p_name, p in self.model.named_parameters():
                    self.second[p_name] = self.beta * self.second[p_name] + (1 - self.beta) * ((p.grad) ** 2)
                    p.grad = torch.div(p.grad, (torch.sqrt(self.second[p_name]) + self.epsilon))

                self.optimizer.step()

