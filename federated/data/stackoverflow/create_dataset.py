"""Data loader for Stackoverflow tag prediction tasks."""

import collections
from typing import Callable, List, Tuple
import json

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


# source: https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/simulation/baselines/stackoverflow/tag_prediction_preprocessing.py

def create_word_vocab(vocab_size: int) -> List[str]:
    """Creates a vocab from the `vocab_size` most common words in Stack Overflow."""
    vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts()
    return list(vocab_dict.keys())[:vocab_size]


def create_tag_vocab(vocab_size: int) -> List[str]:
    """Creates a vocab from the `vocab_size` most common tags in Stack Overflow."""
    tag_dict = tff.simulation.datasets.stackoverflow.load_tag_counts()
    return list(tag_dict.keys())[:vocab_size]


def build_to_ids_fn(word_vocab: List[str],
                    tag_vocab: List[str]) -> Callable[[tf.Tensor], tf.Tensor]:
    """Constructs a function mapping examples to sequences of token indices."""
    word_vocab_size = len(word_vocab)
    word_table_values = np.arange(word_vocab_size, dtype=np.int64)
    word_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(word_vocab, word_table_values),
        num_oov_buckets=1)

    tag_vocab_size = len(tag_vocab)
    tag_table_values = np.arange(tag_vocab_size, dtype=np.int64)
    tag_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(tag_vocab, tag_table_values),
        num_oov_buckets=1)

    def to_ids(example):
        """Converts a Stack Overflow example to a bag-of-words/tags format."""
        sentence = tf.strings.join([example['tokens'], example['title']],
                                   separator=' ')
        words = tf.strings.split(sentence)
        tokens = word_table.lookup(words)
        tokens = tf.one_hot(tokens, word_vocab_size + 1)
        tokens = tf.reduce_mean(tokens, axis=0)[:word_vocab_size]

        tags = example['tags']
        tags = tf.strings.split(tags, sep='|')
        tags = tag_table.lookup(tags)
        tags = tf.one_hot(tags, tag_vocab_size + 1)
        tags = tf.reduce_sum(tags, axis=0)[:tag_vocab_size]

        return (tokens, tags)

    return to_ids


def create_preprocess_fn(
        word_vocab: List[str],
        tag_vocab: List[str],
        client_batch_size: int,
        client_epochs_per_round: int,
        max_elements_per_client: int,
        max_shuffle_buffer_size: int = 10000) -> tff.Computation:
    """Creates a preprocessing function for Stack Overflow tag prediction data.
    This function creates a `tff.Computation` which takes a dataset, and returns
    a preprocessed dataset. This preprocessing takes a maximum number of elements
    in the client's dataset, shuffles, repeats some number of times, and then
    maps the elements to tuples of the form (tokens, tags), where tokens are
    bag-of-words vectors, and tags are binary vectors indicating that a given
    tag is associated with the example.
    Args:
      word_vocab: A list of strings representing the in-vocabulary words.
      tag_vocab: A list of tokens representing the in-vocabulary tags.
      client_batch_size: Integer representing batch size to use on the clients.
      client_epochs_per_round: Number of epochs for which to repeat train client
        dataset. Must be a positive integer.
      max_elements_per_client: Integer controlling the maximum number of elements
        to take per client. If -1, keeps all elements for each client. This is
        applied before repeating `client_epochs_per_round`, and is intended
        primarily to contend with the small set of clients with tens of thousands
        of examples.
      max_shuffle_buffer_size: Maximum shuffle buffer size.
    Returns:
      A `tff.Computation` taking as input a `tf.data.Dataset`, and returning a
      `tf.data.Dataset` formed by preprocessing according to the input arguments.
    """
    if client_batch_size <= 0:
        raise ValueError('client_batch_size must be a positive integer. You have '
                         'passed {}.'.format(client_batch_size))
    elif client_epochs_per_round <= 0:
        raise ValueError('client_epochs_per_round must be a positive integer. '
                         'You have passed {}.'.format(client_epochs_per_round))
    elif max_elements_per_client == 0 or max_elements_per_client < -1:
        raise ValueError(
            'max_elements_per_client must be a positive integer or -1. You have '
            'passed {}.'.format(max_elements_per_client))

    if (max_elements_per_client == -1 or
            max_elements_per_client > max_shuffle_buffer_size):
        shuffle_buffer_size = max_shuffle_buffer_size
    else:
        shuffle_buffer_size = max_elements_per_client

    feature_dtypes = collections.OrderedDict(
        creation_date=tf.string,
        #   title=tf.string,
        score=tf.int64,
        tags=tf.string,
        title=tf.string,
        tokens=tf.string,
        type=tf.string,
    )

    @tff.tf_computation(tff.SequenceType(feature_dtypes))
    def preprocess_fn(dataset):
        to_ids = build_to_ids_fn(word_vocab, tag_vocab)
        return (dataset.take(max_elements_per_client).shuffle(shuffle_buffer_size)
                # Repeat for multiple local client epochs
                .repeat(client_epochs_per_round)
                # Map sentences to tokenized vectors
                .map(to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                # Batch to the given client batch size
                .batch(client_batch_size))

    return preprocess_fn

def get_datasets(
        word_vocab_size=10000,
        tag_vocab_size=500,
        train_batch_size: int = 10000,
        validation_batch_size: int = 500,
        test_batch_size: int = 500,
        num_validation_examples: int = 10000,
        train_shuffle_buffer_size: int = 10000,
        validation_shuffle_buffer_size: int = 1,
        test_shuffle_buffer_size: int = 1
    ) -> tf.data.Dataset:
    """Creates datasets for Stack Overflow tag prediction.
    Args:
      word_vocab_size: Integer representing size of the word vocabulary to use
        when converting sentences to bag-of-words vectors. The word vocabulary
        will consist of the `word_vocab_size` most frequent words in the Stack
        Overflow dataset.
      tag_vocab_size: Integer representing size of the tag vocabulary to use when
        converting lists of tags to bag-of-tags vectors. The tag vocabulary will
        consist of the `tag_vocab_size` most frequent tags in the Stack Overflow
        dataset.
      train_batch_size: The batch size for the training dataset.
      validation_batch_size: The batch size for the validation dataset.
      test_batch_size: The batch size for the test dataset.
      num_validation_examples: Number of examples from Stackoverflow test set to
        use for validation on each round.
      train_shuffle_buffer_size: The shuffle buffer size for the training dataset.
        If set to a number <= 1, no shuffling occurs.
      validation_shuffle_buffer_size: The shuffle buffer size for the validation
        dataset. If set to a number <= 1, no shuffling occurs.
      test_shuffle_buffer_size: The shuffle buffer size for the training dataset.
        If set to a number <= 1, no shuffling occurs.
    Returns:
      train_dataset: A `tf.data.Dataset` instance representing the training
        dataset.
      validation_dataset: A `tf.data.Dataset` instance representing the validation
        dataset.
      test_dataset: A `tf.data.Dataset` instance representing the test dataset.
    """
    if train_shuffle_buffer_size <= 1:
        train_shuffle_buffer_size = 1
    if test_shuffle_buffer_size <= 1:
        test_shuffle_buffer_size = 1

    word_vocab = create_word_vocab(word_vocab_size)
    tag_vocab = create_tag_vocab(tag_vocab_size)
    print(tag_vocab, len(tag_vocab))

    train_preprocess_fn = create_preprocess_fn(
        word_vocab=word_vocab,
        tag_vocab=tag_vocab,
        client_batch_size=train_batch_size,
        client_epochs_per_round=1,
        max_elements_per_client=-1,
        max_shuffle_buffer_size=train_shuffle_buffer_size)

    # validation_preprocess_fn = create_preprocess_fn(
    #     word_vocab=word_vocab,
    #     tag_vocab=tag_vocab,
    #     client_batch_size=validation_batch_size,
    #     client_epochs_per_round=1,
    #     max_elements_per_client=-1,
    #     max_shuffle_buffer_size=validation_shuffle_buffer_size)
    #
    # test_preprocess_fn = create_preprocess_fn(
    #     word_vocab=word_vocab,
    #     tag_vocab=tag_vocab,
    #     client_batch_size=test_batch_size,
    #     client_epochs_per_round=1,
    #     max_elements_per_client=-1,
    #     max_shuffle_buffer_size=test_shuffle_buffer_size)

    raw_train, _, raw_test = tff.simulation.datasets.stackoverflow.load_data()
    train_client_ids = raw_train.client_ids
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    train_path = './data/train/train_client400.json'
    test_path = './data/test/test_client400.json'
    for id in range(400):
        print(id)
        data = raw_train.create_tf_dataset_for_client(train_client_ids[id])
        preprocessed_stackoverflow_train = train_preprocess_fn(data)
        for element in preprocessed_stackoverflow_train:
            preprocess_x = element[0].numpy()
            preprocess_y = element[1].numpy().argmax(1)
        split_point = int(len(preprocess_y) * 4.0 / 5.0)
        preprocess_data_in_train = {
            train_client_ids[id]: {'x': preprocess_x[:split_point].tolist(), 'y': preprocess_y[:split_point].tolist()}}
        preprocess_data_in_test = {
            train_client_ids[id]: {'x': preprocess_x[split_point:].tolist(), 'y': preprocess_y[split_point:].tolist()}}
        uname = train_client_ids[id]
        train_data['users'].append(uname)
        train_data['user_data'].update(preprocess_data_in_train)
        train_data['num_samples'].append(len(preprocess_y[:split_point]))

        test_data['users'].append(uname)
        test_data['user_data'].update(preprocess_data_in_test)
        test_data['num_samples'].append(len(preprocess_y[split_point:]))

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)

get_datasets()