import numpy as np
from tqdm import trange, tqdm
import copy

import tensorflow as tf

from flearn.privacy_analysis.compute_privacy_sgm import *
from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, l2_clip


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using DP Federated Adam to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.m = []
        self.v = []
        for params in self.latest_model:
            self.m.append(np.zeros_like(params))
            self.v.append(np.zeros_like(params))
        self.delta = 1.0 / len(self.clients)

    def train(self):

        print('Training with {} workers ---'.format(self.clients_per_round))

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                #stats_train = self.train_error_and_loss()

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  #  testing accuracy
                #tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                #tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))
                compute_dp_sgd_privacy(len(self.clients), self.clients_per_round, self.sigma,
                                       self.clients_per_round * i / len(self.clients), self.delta)

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)

            csolns = []  # buffer for receiving client solutions

            weights_before = copy.deepcopy(self.latest_model)

            for idx, c in enumerate(selected_clients):  # simply drop the slow devices

                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln = c.solve_inner(self.scale, num_epochs=self.num_epochs, batch_size=self.batch_size, lr=self.learning_rate)
                updates = [(u - v) * 1.0 for u, v in zip(soln, weights_before)]

                l2_norm, clipped = l2_clip(updates, self.clipping_bound)
                csolns.append(clipped)

            sum_updates = self.simple_sum(csolns)
            noise = []
            for p in sum_updates:
                noise.append(np.random.normal(0, self.sigma * self.clipping_bound, size=p.shape))
            averaged_updates = [(u + v) / self.clients_per_round for u, v in zip(sum_updates, noise)]

            adam_updates = []
            for layer, params in enumerate(averaged_updates):
                self.m[layer] = self.adam_beta_1 * self.m[layer] + (1-self.adam_beta_1) * params
                self.v[layer] = self.adam_beta_2 * self.v[layer] + (1-self.adam_beta_2) * np.square(params)
                m_hat = self.m[layer] / (1 - np.power(self.adam_beta_1, i + 1))
                v_hat = self.v[layer] / (1 - np.power(self.adam_beta_2, i + 1))
                adam_updates.append(np.divide(m_hat, np.sqrt(v_hat) + self.epsilon))

            # update models
            self.latest_model = [u+v for u, v in zip(weights_before, adam_updates)]

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
