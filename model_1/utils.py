from typing import Sequence
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import zipfile
import os
import urllib.request as req
from langdetect import detect
import nltk
from nltk.corpus import stopwords
import wandb
from train import train_model
from models import LSTM_current_model

def train_log(loss, example_ct, epoch):
  """Logga podatke na wandb"""
  loss = float(loss)
  wandb.log({"epoch": epoch, "val_loss": loss}, step=example_ct)

def current_accuracy(y_hat, y):
  """Funkcija preciznosti"""
  pred = torch.max(y_hat, 1)[1]
  correct = (pred == y).float().sum().detach().cpu().numpy()
  return correct

def current_criterion(y_hat, y):
  """Loss funkcija"""
  return F.cross_entropy(y_hat, y)

def y_to_device(y,device):
  """Funkcija koja prebacuje y na uredaj"""
  return y.long().to(device)

def model_pipeline(words, train_ds, valid_ds, default_config):
  """Funkcija koja stvara model i pokrece treniranje
  
  Arguments:
    words (`Set[str]`):
      Sve rijeci u vokabularu.
    train_ds, valid_ds (TextDataset):
      Trening i test dataset.
    default_config (`Dict[str,:obj]`):
      Metaparametri za model i treniranje

  Returns:
    :obj : Vraca trenirani model
  """
  global current_criterion, current_accuracy, y_to_device
  vocab_size = len(words)

  with wandb.init(project="praksa", config=default_config):
    config = wandb.config

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=config.batch_size)

    model = LSTM_current_model(len(words), config.embedding_dim, config.hidden_dim, config.dropout)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_model(model, train_dl, val_dl, device, current_criterion, current_accuracy, y_to_device, config.epochs, config.learning_rate)

  return model