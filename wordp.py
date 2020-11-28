import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

train_on_gpu = torch.cuda.is_available()
model_name = 'LSTM_word_pred_20_epoch.net'


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
    return encoded, chars


# ========Encoder========================================


def one_hot_encoder(array, n_labels):
    """:param  array : Array of integers
       :param  n_labels: Total number of lables
       :return: one hot encoded array"""
    one_hot = np.zeros((np.multiply(*array.shape), n_labels), dtype=np.float32)
    # Fill with ones
    one_hot[np.arange(one_hot.shape[0]), array.flatten()] = 1
    # reshape to get back original array
    one_hot = one_hot.reshape((*array.shape, n_labels))
    return one_hot


# =====================================Get the batches from notebook


def get_batches(array, batch_size, sequence_length):
    """Create a generator that returns batches of size
        batch_size x seq_length from arr
       :param array: the encoded array
       :param batch_size: Batch size
       :param sequence_length Sequences in  a batch
       :returns batch_seq array constructed in batch order"""
    batch_size_total = batch_size * sequence_length
    # total number of batches we can make
    n_batches = len(array) // batch_size_total

    # Discard the characters that are not fit with in the batch structure
    array = array[:n_batches * batch_size_total]
    # reshape
    array = array.reshape((batch_size, -1))
    # Iterate through array one sequence at a time
    for n in range(0, array.shape[1], sequence_length):
        # Extract features
        x = array[:, n:n + sequence_length]
        # Now targets Shifted by 1
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], array[:, n + sequence_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], array[:, 0]
        yield x, y


# ===============================================================================================


def get_device():
    """:returns device training device
    """
    if train_on_gpu:
        print("Training on GPU")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Training on CPU")
    return device


######################################################################################################


class CharPredict(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_probability=0.5, lr=0.001):
        super().__init__()
        self.drop_probability = drop_probability
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # Create Character dictionary
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        # Define LSTM
        self.LSTM = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_probability, batch_first=True)
        self.dropout = nn.Dropout(drop_probability)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        """
        :param x : Input
        :param hidden: hidden states
        :returns feed fw o/p and hidden nw
        """
        r_output, hidden = self.LSTM(x, hidden)
        out = self.dropout(r_output)
        # Stack up LSTM's output using view
        out = out.view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        """ Initializes Hidden States"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


# ##########################################################################################


def train(model, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_fraction=0.1, print_every=10):
    """:param model: The RNN LSTM model
       :param data: Text data to train the network
       :param epochs: Number of training cycles
       :param batch_size: Number of mini sequences per mini match
       :param seq_length: Number of characters per batch
       :param lr: Learning Rate
       :param clip: Gradient Clipping Factor
       :param val_fraction: Validation Fraction
       :param print_every: Number of steps for printing training and validation loss"""
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Create Training and validation data
    val_idx = int(len(data) * (1 - val_fraction))
    data, val_data = data[:val_idx], data[val_idx:]
    if train_on_gpu:
        model.cuda()
    counter = 0
    n_chars = len(model.chars)
    for e in range(epochs):
        # initialize Hidden State
        h = model.init_hidden(batch_size)
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            # Encode Data
            x = one_hot_encoder(x, n_chars)
            y = one_hot_encoder(y, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Create new variables for each hidden state
            h = tuple([each.data for each in h])
            # clear the gradient for fresh epoch
            model.zero_grad()
            output, h = model(inputs, h)

            # Calculate loss and perform back propagation
            # t = targets.view(batch_size * seq_length, -1)
            loss = criterion(output, targets.view(batch_size * seq_length, -1))
            loss.backward()
            # Clip of the gradient norms
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                # Get the validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for i, t in get_batches(val_data, batch_size, seq_length):
                    i = one_hot_encoder(i, n_chars)
                    i, t = torch.from_numpy(i), torch.from_numpy(t)
                    val_h = tuple([each.data for each in val_h])
                    val_inputs, val_targets = i, t
                    if train_on_gpu:
                        val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()
                    val_output, val_h = model(val_inputs, val_h)
                    val_loss = criterion(val_output, val_targets.view(batch_size * seq_length))
                    val_losses.append(val_loss.item())
                model.train()

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

    checkpoint = {'n_hidden': model.n_hidden,
                  'n_layers': model.n_layers,
                  'state_dict': model.state_dict(),
                  'tokens': model.chars}
    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)


# ========================================================================================


def predict(model, char, h=None, top_k=None):
    """:param model Network model
       :param char Given Character
       :param h Hidden State
       :param top_k Sampling points
       """
    x = np.array([[model.char2int[char]]])
    x = one_hot_encoder(x, len(model.chars))
    inputs = torch.from_numpy(x)
    if train_on_gpu:
        inputs = inputs.cuda()
    # Detach hidden state from history
    h = tuple([each.data for each in h])
    out, h = model(inputs, h)

    # Get character probabilities
    p = F.softmax(out, dim=1).data
    if train_on_gpu:
        p = p.cpu()
    if top_k is None:
        top_ch = np.arrange(len(model.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeez()
    # Randomly select next set of chars
    p = p.numpy().squeez()
    char = np.random.choice(top_ch, p=p / p.sum())
    return model.int2char[char], h


# ===================================================================


def sample(model, size, prime='The', top_k=None):
    if train_on_gpu:
        model.cuda()
    else:
        model.cpu()
    model.eval()

    # Run through the prime characters
    chars = [ch for ch in prime]
    h = model.init_hidden(1)
    for ch in prime:
        char, h = predict(model, ch, h, top_k=top_k)
        chars.append(char)
    return ''.join(chars)


# ==============================================================================

