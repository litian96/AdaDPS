import numpy as np
import math

import torch
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable

from .model_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import *

class Trainer(BaseTrainer):

    def __init__(self, params):
        super(Trainer, self).__init__(params)
        self.delta = 1 / 60000
        self.mean = dict()
        self.preconditioner = dict()
        for p_name, p in self.model.named_parameters():
            self.mean[p_name] = torch.zeros_like(p)
            self.preconditioner[p_name] = torch.zeros_like(p)


    def train(self):

        total_step = len(self.train_loader)
        total_iter = total_step * self.epochs

        for epoch in range(self.epochs):

            if epoch % self.eval_every_epoch == 0:
                #l_train = self.get_train_loss()
                l_test = self.get_test_loss()
                print('epoch {}, test loss {:.5f}'.format(epoch, l_test))
                # privacy tracking (given the total number of samples, the mini-batch size, sigma, the number of epochs, delta,
                # compute eps based on RDP
                compute_dp_sgd_privacy(total_step * self.batch_size, self.batch_size, \
                                       self.sigma, epoch, self.delta)
                print()



            for i, (images, labels) in enumerate(self.train_loader):

                pub_g = self.get_pub_gradient()

                images = images.reshape(-1, 784)
                images = Variable(images).to(self.device)
                reconstructed = self.model(images)
                l = self.loss_flat(reconstructed, images)

                if len(l) % self.num_microbatches == 0:  
                    # after reshape: (16, 4 * 784)
                    # after mean: (16,)
                    losses = torch.mean(l.reshape(self.num_microbatches, -1), dim=1)
                else:
                    continue
                saved_var = dict()
                for p_name, p in self.model.named_parameters():
                    saved_var[p_name] = torch.zeros_like(p)

                for j in losses:  # for every micro-batch in the mini-batch
                    self.model.zero_grad()
                    j.backward(retain_graph=True)

                    norm_before = clip_grad_norm_(self.model.parameters(), self.clipping_bound)
                    if i % 500 == 0:
                        print(i, 'norm: ', norm_before)

                    for p_name, p in self.model.named_parameters():
                        new_grad = p.grad
                        saved_var[p_name].add_(new_grad)

                global_iter_num = epoch * total_step + i
                alpha_t = np.cos(math.pi * global_iter_num / (2 * total_iter))

                if i % 100 == 0:
                    print(epoch, i, alpha_t)

                for p_name, p in self.model.named_parameters():
                    if self.device.type == 'cuda':
                        noise = torch.cuda.FloatTensor(p.grad.shape).normal_(0, self.sigma * self.clipping_bound)
                    else:
                        noise = torch.FloatTensor(p.grad.shape).normal_(0, self.sigma * self.clipping_bound)
                    saved_var[p_name].add_(noise)
                    p.grad = saved_var[p_name] / self.num_microbatches

                    p.grad = saved_var[p_name] / self.num_microbatches
                    p.grad = alpha_t * p.grad + (1 - alpha_t) * pub_g[p_name]



                self.optimizer.step()







