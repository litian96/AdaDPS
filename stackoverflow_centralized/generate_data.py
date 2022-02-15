import numpy as np
import random
import os

def generate_stackoverflow():
    train_file_path = './data/train'
    test_file_path = './data/test'
    base_test_dir = '../federated/data/stackoverflow/data/test/test_np'
    base_train_dir = '../federated/data/stackoverflow/data/train/train_np'
    train_samples = {'x': [], 'y': []}
    test_samples = {'x': [], 'y': []}
    for i, c in enumerate(os.listdir(base_test_dir)):
        if i % 100 == 0: print(i)
        test_c = np.load(os.path.join(base_dir, c))
        test_samples['x'].append(test_c['x'])
        test_samples['y'].append(test_c['y'])
    test_samples['x'] = np.concatenate(test_samples['x'], axis=0)
    test_samples['y'] = np.concatenate(test_samples['y'])
    np.random.seed(123)
    perm = np.random.permutation(len(test_samples['y']))
    test_samples['x'], test_samples['y'] = test_samples['x'][perm], test_samples['y'][perm]
    print('begin saving test')
    np.savez(test_file_path, x=test_samples['x'], y=test_samples['y'])
    print('finished')


    for i, c in enumerate(os.listdir(base_train_dir)):
        if i % 100 == 0: print(i)
        train_c = np.load(os.path.join(base_train_dir, c))
        train_samples['x'].append(train_c['x'])
        train_samples['y'].append(train_c['y'])
    train_samples['x'] = np.concatenate(train_samples['x'], axis=0)
    train_samples['y'] = np.concatenate(train_samples['y'])
    np.random.seed(123)
    perm = np.random.permutation(len(train_samples['y']))
    train_samples['x'], train_samples['y'] = train_samples['x'][perm], train_samples['y'][perm]
    print('begin saving train')
    np.savez(train_file_path, x=train_samples['x'], y=train_samples['y'])
    print('finished')


generate_stackoverflow()