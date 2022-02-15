import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from .model_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import *

import copy

class Trainer(BaseTrainer):

    def __init__(self, params):
        super(Trainer, self).__init__(params)
        self.delta = 1 / 240000
        self.mean=dict()
        self.preconditioner = dict()
        for p_name, p in self.model.named_parameters():
            self.mean[p_name] = torch.zeros_like(p)
            self.preconditioner[p_name] = torch.zeros_like(p)

    def train(self):

        total_step = len(self.train_loader)

        # freq = np.load('diagnostics/frequency.npy')
        # freq = np.where(freq < 1e-20, np.average(freq[9000:10000]), freq)
        # multiplier = np.minimum(self.multiplier_cap, np.divide(1.0, freq) / 25)
        # multiplier = torch.FloatTensor(multiplier).cuda()

        for epoch in range(self.epochs):

            if self.use_public:
                self.estimate_preconditioner()

            if epoch % self.eval_every_epoch == 0:
                # l = self.get_train_loss()
                accu = self.get_test_accuracy()
                print('epoch {}, test accuracy {:.5f}'.format(epoch, accu))
                # privacy tracking (given the total number of samples, the mini-batch size, sigma, the number of epochs, delta,
                # compute eps based on RDP
                compute_dp_sgd_privacy(total_step * self.batch_size, self.batch_size, self.sigma, epoch, self.delta)
                print()
                self.model.train()  # turn on training

            for i, (x, labels) in enumerate(self.train_loader):

                x = x.to(self.device)
                labels = labels.to(self.device)

                predicted = self.model(x)
                l = self.loss_flat(predicted, labels)
                if len(l) % self.num_microbatches == 0:
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
                                p.grad[:, 100:] = p.grad[:, 100:] * 20

                    if self.use_public:
                        for p_name, p in self.model.named_parameters():
                            #p.grad = 0.9 * p.grad + 0.1 * self.mean[p_name]  # momentum for adam
                            p.grad = p.grad / torch.sqrt(self.preconditioner[p_name]+1e-5)

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






