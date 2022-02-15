import numpy as np
import copy
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        updates_mom = []

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0 and i > 0:
                stats = self.test()  # have set the latest model for all clients
                #stats_train = self.train_error_and_loss()
                if i == 0: print('total training samples: ', np.sum(stats[2]))
                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                #tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                #tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)

            csolns = []  # buffer for receiving client solutions

            weights_before = copy.deepcopy(self.latest_model)
            cupdates = []

            for idx, c in enumerate(selected_clients):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln = c.solve_inner(self.scale, num_epochs=self.num_epochs, batch_size=self.batch_size, lr=self.learning_rate)

                updates = [(u - v) * 1.0 for u, v in zip(soln, weights_before)]

                cupdates.append(updates)

                # gather solutions from client
                csolns.append(soln)

            overall_updates = self.simple_aggregate(cupdates)

            # optional: momentum
            updates_mom = [(0.9 * u + 0.1 * v) for u, v in zip(updates_mom, overall_updates)]

            #self.latest_model = self.simple_aggregate(csolns)
            self.latest_model = [(u+v) for u, v in zip(weights_before, updates_mom)]


        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
