import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import utils
from collections import Counter
import random

train_on_gpu = torch.cuda.is_available()
model_name = 'LSTM_word_pred_20_epoch.net'


# ===========================================================
# Load data and convert tot integers


def load_data(file_path):
    """:param file_path : the file path
        :returns : the parsed text
    """
    with open(file_path) as f:
        text = f.read()
    return text


# ======================================================


def preprocess(text):
    """:param: text: Incoming string of words
       :return: train_words: encoded integer stream"""
    words = utils.preprocess(text)
    vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]
    # Remove Noise words
    threshold = 1e-5
    word_counts = Counter(int_words)
    total_count = len(int_words)
    freq = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold / freq[word]) for word in word_counts}
    # Discard the very frequent words as per the subsampling equation. Refer readme for this explanation
    train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]
    return train_words


# ========================================================================================


def get_target(words, idx, window_size=5):
    """:param words: The incoming list of words
       :param idx : The pivot index
       :param window_size: The number of past and future words to select
        Function selects R number of words from past and furure, with in the specified
        window size
        :returns """

    random_size = np.random.randint(1, window_size + 1)
    start = idx - random_size if (idx - random_size) > 0 else 0
    stop = idx + random_size
    target_words = words[start: idx] + words[idx + 1: stop + 1]
    return list(target_words)


# ---------------------------------------------------------------------------------------

def get_batches(words, batch_size, window_size=5):
    """:param words: The incominsg List of words
       :param batch_size: the chosen batch size
       :param window_size: The number of words to be fetched from past and future
       """
    number_batches = len(words) // batch_size
    words = words[:number_batches * batch_size]
    for index in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[index:index + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y


# ========Similarity========================================


def similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """
    Returns Cosine Similarity of the validation words with words in the embedding matrix
    :param embedding: The embedding matrix
    :param valid_size: Validation Size
    :param valid_window: The window of words around the current words
    :param device: Defaulted to CPU
    """
    embed_vectors = embedding.weight
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqeeze(0)
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, random.sample(range(1000, 1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t()) / magnitudes
    return valid_examples, similarities

# ____________________________________________Define the model


class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()

        self.embed = nn.Embedding(n_vocab, n_embed)
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        scores = self.output(x)
        log_ps = self.log_softmax(scores)
        return log_ps

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Training





