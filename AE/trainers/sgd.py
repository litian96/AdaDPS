import numpy as np

import torch
from torch.autograd import Variable
import copy

from .model_base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, params):
        super(Trainer, self).__init__(params)
        self.preconditioner = dict()
        self.mean = dict()

        for p_name, p in self.model.named_parameters():
            self.preconditioner[p_name] = torch.zeros_like(p)
            self.mean[p_name] = torch.zeros_like(p)

    def train(self):
        total_step = len(self.train_loader)

        freq = self.get_feature_frequency()
        multiplier = np.minimum(self.multiplier_cap, np.divide(max(freq), freq+1e-10))
        multiplier = torch.FloatTensor(multiplier).cuda()


        for epoch in range(self.epochs):

            if epoch % self.eval_every_epoch == 0:
                l_train = self.get_train_loss()
                l_test = self.get_test_loss()
                print('epoch {}, test loss {:.5f}, training loss {:.5f}'.format(epoch, l_test, l_train))

            if self.use_public:
                self.estimate_preconditioner()

            for i, (images, labels) in enumerate(self.train_loader):
                images = images.reshape(-1, 784)
                images = Variable(images).to(self.device)
                reconstructed = self.model(images)
                l = self.loss(reconstructed, images)
                self.model.zero_grad()
                l.backward()

                # disable momentum by default
                if self.momentum:
                    for p_name, p in self.model.named_parameters():
                        self.mean[p_name] = self.momentum_parameter * self.mean[p_name] + (1-self.momentum_parameter) * p.grad
                        p.grad = copy.deepcopy(self.mean[p_name])

                if self.use_public:
                    for p_name, p in self.model.named_parameters():
                        p.grad = p.grad / torch.sqrt(self.preconditioner[p_name]+self.epsilon)


                self.optimizer.step()  # apply p.grad




