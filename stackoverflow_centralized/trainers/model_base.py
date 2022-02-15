import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import *
import copy


class BaseTrainer(object):
    def __init__(self, params):

        for key, val in params.items():
            setattr(self, key, val)

        train_data = np.load('data/train.npz')
        test_data = np.load('data/test.npz')

        x_train, y_train = train_data['x'].astype(np.float32), train_data['y']
        perm = np.random.permutation(len(y_train))
        x_train, y_train = x_train[perm], y_train[perm]  # randomly shuffle
        x_public, y_public = x_train[:2400], y_train[:2400]
        x_train, y_train = x_train[2400:], y_train[2400:]
        x_test, y_test = test_data['x'].astype(np.float32), test_data['y']
        print('x train: ', x_train.shape)  # x train:  (246092, 10000) [all training data]
        print('x test: ', x_test.shape)

        self.public_dataset = TensorDataset(torch.FloatTensor(x_public), torch.LongTensor(y_public))
        self.train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
        self.test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))

        self.public_x = torch.FloatTensor(x_public)
        self.public_y = torch.LongTensor(y_public)


        self.public_loader = DataLoader(dataset=self.public_dataset,
                                        batch_size=self.public_bs,
                                        num_workers=4,
                                        shuffle=True)

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        num_workers=4,
                                                        shuffle=True)

        self.test_loader = DataLoader(dataset=self.test_dataset,
                                                       batch_size=self.batch_size,
                                                       num_workers=4,
                                                       shuffle=False)

        self.model = nn.Linear(10000, 500)  # around 5 million model parameters in total

        self.loss = nn.CrossEntropyLoss()
        self.loss_flat = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)


    def estimate_preconditioner(self):

        tmp_mean = dict()

        for p_name, p in self.model.named_parameters():
            self.preconditioner[p_name] = torch.zeros_like(p)
            tmp_mean[p_name] = torch.zeros_like(p)

        for i, (x_pub, labels_pub) in enumerate(self.public_loader):
            x_pub = x_pub.to(self.device)
            labels_pub = labels_pub.to(self.device)
            predicted = self.model(x_pub)
            l = self.loss(predicted, labels_pub)
            self.model.zero_grad()
            l.backward()
            for p_name, p in self.model.named_parameters():
                self.preconditioner[p_name] += p.grad ** 2
                tmp_mean[p_name] += p.grad


        # optional: didn't estimate mean on public data for exps in the main text (RMSPROP+AdaDPS)
        # only updated self.preconditioner

        for p_name, p in self.model.named_parameters():
            self.preconditioner[p_name] = self.preconditioner[p_name] / (i+1)
            self.mean[p_name] = 0.9 * self.mean[p_name] + 0.1 * (tmp_mean[p_name] / (i+1))


    def get_pub_gradient(self):
        pub_g = dict()

        for p_name, p in self.model.named_parameters():
            pub_g[p_name] = torch.zeros_like(p)

        i = np.random.choice(range(2400-self.public_bs))

        x_pub = self.public_x[i:i+self.public_bs].to(self.device)
        labels_pub = self.public_y[i:i+self.public_bs].to(self.device)
        predicted = self.model(x_pub)
        l = self.loss(predicted, labels_pub)
        self.model.zero_grad()
        l.backward()
        for p_name, p in self.model.named_parameters():
            pub_g[p_name] = copy.deepcopy(p.grad)
        return pub_g


    def get_loss_and_gradients(self, input, labels):
        predicted = self.model(input)
        l = self.loss(predicted, labels)
        self.optimizer.zero_grad()
        l.backward()
        g = []
        for x in self.model.parameters():
            g.append(x.grad)
        return l.item(), g

    def apply_gradients(self, grads):
        for i, x in enumerate(self.model.parameters()):
            x.grad.data = grads[i]
        self.optimizer.step()

    def get_gradient_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            total_norm += p.grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def get_l2_norm(self, g):
        total_norm = 0
        for p in g:
            total_norm += p.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def get_test_accuracy(self):
        self.model.eval()
        with torch.no_grad():
            correct=0
            total=0
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().data
        return correct*1.0 / total

    def get_train_loss(self):
        self.model.eval()
        with torch.no_grad():  # not optimizing
            loss = 0
            for i, (x, labels) in enumerate(self.train_loader):
                x = x.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(x)
                l = self.loss(outputs, labels)
                loss += l.item()
        return loss/i



