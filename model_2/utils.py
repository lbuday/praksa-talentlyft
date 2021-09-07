import torch
from langdetect import detect
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class BertClassificationCollator(object):
    """Koristi dani tokenizer i label enkoder da pretvori text i oznake u brojeve
    koji moguici direktno u Bert model.

    Arguments:
      use_tokenizer (:obj:`transformers.tokenization_?`):
          Tokenizator kojeg koristimo.

      labels_ids (:obj:`dict`):
          Rijecnik koji pretvara oznake u brojeve.

      max_sequence_len (:obj:`int`, `optional`)
          Maksimalna zeljena duljina inputa, podrezuje dulje
          i padda krace.

    """
    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        #provjera maksimalne duljine
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder
        return

    def __call__(self, sequences):
        """Funkcija koja dozvoljava da se klasa koristi kao funkcija

        Arguments:

          item (:obj:`list`):
              List of tekstova i oznaka.

        Returns:
          :obj:`Dict[str, object]`: Rijecnik ulaza koje dajemo modelu.
        """
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        labels = [self.labels_encoder[label] for label in labels]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        inputs.update({'labels':torch.tensor(labels)})

        return inputs