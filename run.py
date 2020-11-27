import wordp

file_path1 = 'anna.txt'
n_hidden_states = 512
n_layers = 2
n_epoch = 20
n_batch_size = 128
n_seq = 100
file_text = wordp.load_data(file_path1)
encoded_text, char_set = wordp.map_char_to_int(file_text)
Rnn_Model = wordp.CharPredict(char_set, n_hidden_states, n_layers)
wordp.train(Rnn_Model, encoded_text, epochs=n_epoch, batch_size=n_batch_size, seq_length=n_seq)
print(wordp.sample(Rnn_Model, 1000, prime='Anna', top_k=5))
