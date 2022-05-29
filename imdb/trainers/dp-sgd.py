import numpy as np
import copy

import torch
from torch.nn.utils import clip_grad_norm_
from .model_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import *


class Trainer(BaseTrainer):

    def __init__(self, params):
        super(Trainer, self).__init__(params)
        self.delta = 1.0 / 25000
        self.mean = dict()
        self.preconditioner = dict()
        self.beta = 0.9
        for p_name, p in self.model.named_parameters():
            self.mean[p_name] = torch.zeros_like(p)
            self.preconditioner[p_name] = torch.zeros_like(p)

        print('total number of parameters: ', sum(p.numel() for p in self.model.parameters()))

    def train(self):

        total_step = len(self.train_loader)

        if self.scale_by == 'tfidf':
            freq = self.get_tf_idf_value()
            multiplier = freq * 10
            multiplier[:3] = 0.01
        elif self.scale_by == 'freq':
            freq = self.get_bow_frequency()
            multiplier = np.minimum(self.multiplier_cap, np.divide(max(freq), freq + 1e-20)) / self.division
        multiplier = torch.FloatTensor(multiplier).cuda()

        for epoch in range(self.epochs):

            if self.use_public:
                self.estimate_preconditioner()

            if epoch % self.eval_every_epoch == 0:
                train_accu, train_l = self.get_train_accuracy_and_loss()
                accu = self.get_test_accuracy()
                print('epoch {}, test accuracy {:.5f}, training accuracy {:.5f}, training loss {:.5f}' \
                      .format(epoch, accu, train_accu, train_l))
                # compute eps based on RDP (should use self.noise)
                compute_dp_sgd_privacy(total_step * self.batch_size, self.batch_size, self.sigma, epoch, self.delta)
                print()
                self.model.train()  # turn on training

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

                    if self.scale:
                        for p_name, p in self.model.named_parameters():
                            if p_name == 'weight':
                                p.grad = p.grad * multiplier

                    if self.use_public:
                        for p_name, p in self.model.named_parameters():
                            # p.grad = 0.9 * p.grad + 0.1 * self.mean[p_name]  # momentum, optional
                            p.grad = p.grad / torch.sqrt(self.preconditioner[p_name]+self.epsilon)

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

                self.optimizer.step()











