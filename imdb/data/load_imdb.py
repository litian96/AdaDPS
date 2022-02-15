import numpy as np
import json
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict

TOP_WORDS = 10000
MAX_LEN = 290

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=TOP_WORDS, skip_top=0)

word_index = imdb.get_word_index()
word_index_sorted = OrderedDict(sorted(word_index.items(), key=lambda item: item[1]))
with open("word_index.txt", "w") as file:
    for k, v in word_index_sorted.items():
        a = str(v+3) + ": " + str(k.encode('utf-8')) + "\n"
        file.write(a)

###### confirm the word and indexing
#
# (x_train, _), _ = imdb.load_data()
# # Retrieve the word index file mapping words to indices
# word_index = imdb.get_word_index()
# # Reverse the word index to obtain a dict mapping indices to words
# ### attention here... otherwise, cannot restore the original text
# ### which means that token ids in input features
# word_index = {k:(v+3) for k,v in word_index.items()}
# word_index["<PAD>"] = 0
# word_index["<START>"] = 1
# word_index["<UNK>"] = 2
# word_index["<UNUSED>"] = 3
# # Decode the first sequence in the dataset
# inverted_word_index = dict((i, word) for (word, i) in word_index.items())
# decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
# print(decoded_sequence)
#
# x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
# x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
# print(x_train.shape, x_test.shape)

#

# np.savez('imdb_10000d_train', x=x_train, y=y_train)
# np.savez('imdb_10000d_test', x=x_test, y=y_test)
#data = np.load('imdb_2500d_train.npz')
#print(data['x'][0])