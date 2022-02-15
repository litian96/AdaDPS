import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = np.load('imdb_10000d_train.npz')
test_data = np.load('imdb_10000d_test.npz')



sample_len = len(train_data['x'][0])
num_vocab = 10000
num_sample = len(train_data['y'])
train_x, train_y = train_data['x'], train_data['y']
test_x, test_y = test_data['x'], test_data['y']

################## TF-IDF ###################
#
#
# tf_idf_train = np.zeros((num_sample, num_vocab))
# tf_idf_test = np.zeros((num_sample, num_vocab))
# print(train_x.shape, test_x.shape)
#
# token_in_sample = np.zeros((num_sample*2, num_vocab))  # if sample i has token j
# token_in_sample_count = np.zeros((num_sample*2, num_vocab))  # sample i has how much percentage of token j
#
#
# for i, x in enumerate(train_x):
#     for token_id in x:
#         token_in_sample[i][token_id] = 1
#         token_in_sample_count[i][token_id] += 1
#
# for i, x in enumerate(test_x):
#     for token_id in x:
#         token_in_sample[i+len(train_x)][token_id] = 1
#         token_in_sample_count[i+len(train_x)][token_id] += 1
#
# token_in_how_many_samples = np.sum(token_in_sample, axis=0)
# token_in_how_many_samples = np.where(token_in_how_many_samples == 0, token_in_how_many_samples+0.1, token_in_how_many_samples)
#
#
# for i, x in enumerate(train_x):
#     token_freq = token_in_sample_count[i] / len(x)
#     tf_idf_train[i] = np.multiply(token_freq, np.log10(num_sample*2/token_in_how_many_samples))
#
#
# for i, x in enumerate(test_x):
#     token_freq = token_in_sample_count[i+len(train_x)] / len(x)
#     tf_idf_test[i] = np.multiply(token_freq, np.log10(num_sample*2/token_in_how_many_samples))
#
# np.savez('imdb_10000d_train_tf-idf', x=tf_idf_train, y=train_y)
# np.savez('imdb_10000d_test_tf-idf', x=tf_idf_test, y=test_y)


############## bag of words #####################
bow_train = np.zeros((num_sample, num_vocab))
bow_test = np.zeros((num_sample, num_vocab))


for i, x in enumerate(train_x):
    a = np.zeros(10000)
    for token in x:
        a[token] += 1

    bow_train[i] = a / len(x)  # len(x)=290

for i, x in enumerate(test_x):
    a = np.zeros(10000)
    for token in x:
        a[token] += 1
    bow_test[i] = a / len(x)

np.savez('imdb_10000d_train_bow', x=bow_train, y=train_y)
np.savez('imdb_10000d_test_bow', x=bow_test, y=test_y)