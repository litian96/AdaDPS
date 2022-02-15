import numpy as np
import random
import sklearn.datasets

def generate_mnist():

    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', cache=True)
    print(mnist.data.shape)
    mnist.target = mnist.target.astype(np.int8)
    mnist.data = mnist.data / 255 # scale to [0,1]
    train_x, train_y = mnist.data[:60000], mnist.target[:60000]
    test_x, test_y = mnist.data[60000:], mnist.target[60000:]
    return (train_x, train_y), (test_x, test_y)


def generate_toy():

    np.random.seed(666)

    n, d = 1000, 500
    x = np.zeros((n, d))
    prob = np.zeros(d)
    for j in range(d):
        if j < d/10:
            prob[j] = 0.9
        else:
            prob[j] = 0.01

    for j in range(d):
        x[:, j] = np.random.choice([0, 1], size=n, p=[1-prob[j], prob[j]])

    w = np.zeros(d)
    w[:int(d/10)] = 0.01
    w[int(d/10):] = 1.0
    y = np.dot(x, w) + np.random.normal(0, 0.01, size=n)
    print('len y: ', len(y))

    ground_truth = w
    print(ground_truth)

    perm = np.arange(1000)
    np.random.shuffle(perm)

    new_x, new_y = x[perm], y[perm]
    return new_x, new_y

