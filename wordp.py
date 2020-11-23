import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# ===========================================================
# Load data and convert tot integers


def load_data(file_path):
    """:param file_path : the file path
        :returns : the parsed text
    """
    with open('anna.txt', 'r') as f:
        text = f.read()
    return text
# ======================================================


def map_char_to_int(text):
    """:param: text: Incoming string of words
       :return: encoded integer stream"""
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char_to_int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char_to_int[ch] for ch in text])
    return encoded

# ========Encoder========================================


def one_hot_encoder(array, n_labels):
    """:param  array : Array of integers
       :param  n_labels: Total number of lables
       :return: one hot encoded array"""
    one_hot = np.zeros((np.multiply(*array.shape), n_labels), dtype=np.float32)
    # Fill with ones
    one_hot[np.arange(one_hot.shape[0]), array.flatten()] = 1
    # reshape to get back original array
    one_hot = one_hot.reshape((*array.shape(), n_labels))
    return one_hot

# =====================================
