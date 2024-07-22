from collections import OrderedDict
import warnings
from typing import Optional, Dict, Tuple, List
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size):
        super(CharRNN, self).__init__()
        self.batch_size = batch_size
        self.rnn_units = rnn_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.rnn_units).to(device),
                torch.zeros(1, batch_size, self.rnn_units).to(device))


# Length of the vocabulary in chars.
vocab_size = len(vocab)

# The embedding dimension.
embedding_dim = 256

# Number of RNN units.

rnn_units = 1024
model=CharRNN(vocab_size,embedding_dim,rnn_units,batchi_size=BATCH_SIZE)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)