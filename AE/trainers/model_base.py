import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

# tensorflow implementation:
# https://github.com/google-research/federated/blob/780767fdf68f2f11814d41bbbfe708274eb6d8b3/utils/models/emnist_ae_models.py

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 30)
        self.fc5 = nn.Linear(30, 250)
        self.fc6 = nn.Linear(250, 500)
        self.fc7 = nn.Linear(500, 1000)
        self.fc8 = nn.Linear(1000, 784)

    def forward(self, x):
        x = x.view(-1, 784)  # (batch size, 784)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        x = F.sigmoid(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        x = F.sigmoid(self.fc7(x))
        x = F.sigmoid(self.fc8(x))

        return x  # (batch size, 784)

class BaseTrainer(object):
    def __init__(self, params):
        for key, val in params.items():
            setattr(self, key, val)

        if self.dataset == 'mnist':

            self.train_dataset = torchvision.datasets.MNIST(root='./data',
                                                            train=True,
                                                            transform=transforms.ToTensor(),  # scale to [0,1]
                                                            download=True)
            self.test_dataset = torchvision.datasets.MNIST(root='./data',
                                                           train=False,
                                                           transform=transforms.ToTensor(),
                                                           )

            perm = np.random.permutation(60000)[:600]
            self.public_dataset = torch.utils.data.Subset(self.train_dataset, perm)


        elif self.dataset == 'fashionmnist':

            self.train_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                            train=True,
                                                            transform=transforms.ToTensor(),  # scale to [0,1]
                                                            download=True)
            self.test_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                           train=False,
                                                           transform=transforms.ToTensor(),
                                                           )

        self.public_loader = torch.utils.data.DataLoader(dataset=self.public_dataset,
                                                        batch_size=self.public_bs,
                                                        num_workers=4,
                                                        shuffle=True)

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        num_workers=4,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=self.batch_size,
                                                       num_workers=4,
                                                       shuffle=False)

        self.model = AE()
        #self.loss = nn.BCELoss()
        self.loss = nn.MSELoss()  # mean square loss per pixel
        self.loss_flat = nn.MSELoss(reduction='none')  # size: (batch_size, 784)

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
            x_pub = x_pub.reshape(-1, 784)
            reconstructed = self.model(x_pub)
            l = self.loss(reconstructed, x_pub)
            self.model.zero_grad()
            l.backward()
            for p_name, p in self.model.named_parameters():
                self.preconditioner[p_name] += p.grad ** 2
                tmp_mean[p_name] += p.grad

        for p_name, p in self.model.named_parameters():
            self.preconditioner[p_name] = self.preconditioner[p_name] / (i+1)
            self.mean[p_name] = 0.9 * self.mean[p_name] + 0.1 * (tmp_mean[p_name] / (i+1))


    def get_pub_gradient(self):

        pub_g = dict()

        for p_name, p in self.model.named_parameters():
            pub_g[p_name] = torch.zeros_like(p)

        for i, (x_pub, labels_pub) in enumerate(self.public_loader):
            x_pub = x_pub.to(self.device)
            x_pub = x_pub.reshape(-1, 784)
            reconstructed = self.model(x_pub)
            l = self.loss(reconstructed, x_pub)
            self.model.zero_grad()
            l.backward()
            for p_name, p in self.model.named_parameters():
                pub_g[p_name] = copy.deepcopy(p.grad)
            break  # only run for 1 mini-batch
        return pub_g


    def update_preconditioner(self):

        for i, (x_pub, labels_pub) in enumerate(self.public_loader):
            x_pub = x_pub.to(self.device)
            x_pub = x_pub.reshape(-1, 784)
            reconstructed = self.model(x_pub)
            l = self.loss(reconstructed, x_pub)
            self.model.zero_grad()
            l.backward()
            for p_name, p in self.model.named_parameters():
                self.preconditioner[p_name] = 0.9 * self.preconditioner[p_name] + 0.1 * (p.grad ** 2)


    def get_feature_frequency(self):
        freq = np.zeros(28 * 28)
        #E_g2 = np.zeros(28 * 28)
        idx = 0
        for i, (x, labels) in enumerate(self.train_loader):
            for sample in x:
                freq += np.asarray(sample.flatten())
                #E_g2 += np.asarray(sample.flatten() ** 2)
                idx += 1
        freq = freq / idx
        #E_g2 = E_g2 / idx
        return freq


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

    def get_test_loss(self):
        with torch.no_grad():
            loss = 0
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                images = images.reshape(-1, 784)
                reconstructed = self.model(images)
                l = self.loss(reconstructed, images)
                loss += l.item()
        return loss/i

    def get_train_loss(self):
        with torch.no_grad():  # not optimizing
            loss = 0
            for i, (x, labels) in enumerate(self.train_loader):
                x = x.to(self.device)
                x = x.reshape(-1, 784)
                reconstructed = self.model(x)
                l = self.loss(reconstructed, x)
                loss += l.item()
        return loss/i


