import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM_current_model(torch.nn.Module) :
  """Model sa tri sloja LSTM-a
  
  Arguments:
    vocab_size (`int`):
      Velicina vokabulara.
    embedding_dim (`int`):
      Velicina ugradivanja.
    hidden_dim (`List[int]`):
      Lista intova koji definiraju velicinu,
      svakog sloja u modelu
      (duljina minimalna 3).
    drouput rate (`float`):
      Dropout stopa.
  """
  def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.rnn_1 = nn.LSTM(embedding_dim, hidden_dim[0], batch_first=True)
    self.rnn_2 = nn.LSTM(hidden_dim[0], hidden_dim[1], batch_first=True)
    self.rnn_3 = nn.LSTM(hidden_dim[1], hidden_dim[2], batch_first=True)
    self.linear = nn.Linear(hidden_dim[2], 3)
    self.dropout = nn.Dropout(dropout_rate)
    self.softmax = nn.Softmax(dim=1)
      
  def forward(self, x, l):
    x = self.embeddings(x)
    x = self.dropout(x)
    try:
      x_pack = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
    except RuntimeError:
      print(x,l)
    rnn_out, (ht_1,ct) = self.rnn_1(x_pack)
    rnn_out, (ht_2,ct) = self.rnn_2(rnn_out)
    _, (ht_3,ct) = self.rnn_3(rnn_out)
    return self.softmax(self.linear(ht_3[-1]))