import wordp
import torch
from torch import nn
import torch.optim as optimizer
file_path1 = 'anna.txt'
n_hidden_states = 512
n_layers = 2
n_epoch = 5
n_batch_size = 128
n_seq = 100
embedding_dim = 300
file_text = wordp.load_data(file_path1)
train_words, vocab_to_int, int_to_vocab = wordp.preprocess(file_text)

Rnn_Model = wordp.SkipGram(len(vocab_to_int), embedding_dim)
criterion = nn.NLLLoss()
model_optimizer = optimizer.Adam(Rnn_Model.params(), lr = 0.003)
wordp.train(Rnn_Model, encoded_text, epochs=n_epoch, batch_size=n_batch_size, seq_length=n_seq)
print(wordp.sample(Rnn_Model, 1000, prime='Anna', top_k=5))
