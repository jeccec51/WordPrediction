import wordp
import torch
from torch import nn
import torch.optim as optimizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

file_path1 = 'text8/text8.txt'
n_hidden_states = 512
n_layers = 2
n_epoch = 5
n_batch_size = 128
n_seq = 100
embedding_dim = 300
file_text = wordp.load_data(file_path1)
train_words, vocab_to_int, int_to_vocab = wordp.preprocess(file_text)

Embed_Model = wordp.SkipGram(len(vocab_to_int), embedding_dim)
criterion = nn.NLLLoss()
model_optimizer = optimizer.Adam(Embed_Model.parameters(), lr=0.003)
device = wordp.get_device()
wordp.train_network(Embed_Model, device, criterion, model_optimizer, n_epoch, train_words, int_to_vocab)

embeddings = Embed_Model.embed.weight.to('cpu').data.numpy()
viz_words = 600
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])
fig, ax = plt.subplots(figsize=(16, 16))
for index in range(viz_words):
    plt.scatter(*embed_tsne[index, :], color='steelblue')
    plt.annotate(int_to_vocab[index], (embed_tsne[index, 0], embed_tsne[index, 1]), alpha=0.7)
