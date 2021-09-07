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
from sklearn.model_selection import train_test_split
from data import encode_sentence, tokenize, check_text, TextDataset
from utils import model_pipeline


if __name__ == "__main__":
    df = pd.read_csv("../jobs.csv")

    nltk.download('stopwords')
    en_stops = set(stopwords.words('english'))

    #Provjerava jezik teksta, tokenizira i uklanja stop rijeci
    df['benefits_data'] = df['JbBenefitsSection'].apply(lambda x: tokenize(x, en_stops) if check_text(x) else [])
    print("done")
    df['info_data'] = df['JbInfoSection'].apply(lambda x: tokenize(x, en_stops) if check_text(x) else [])
    print("done")
    df['requirements_data'] = df['JbRequirementsSection'].apply(lambda x: tokenize(x, en_stops) if check_text(x) else [])

    #Spaja sve podatke
    tokenized_list = list(df['benefits_data']) + list(df['info_data']) + list(df['requirements_data'])

    #Broji ponavljivanja odredenih rijeci
    counts = Counter()
    for elem in tqdm(tokenized_list):
      counts.update(elem)

    #Uklanja svaku rijec koja se pojavljuje samo jednom
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]

    #Stvara indekse za svaku rijec
    #Dodani su START i END tokeni za prepoznavanje pocetka i kraja
    vocab2index = {"PAD":0, "UNK":1, "START":2, "END":3}
    words = ["PAD", "UNK", "START", "END"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    #Pretvara recenice stringova u liste indeksa
    df['benefits_data'] = df['benefits_data'].apply(lambda x: np.array(encode_sentence(x,vocab2index)))
    df['info_data'] = df['info_data'].apply(lambda x: np.array(encode_sentence(x,vocab2index)))
    df['requirements_data'] = df['requirements_data'].apply(lambda x: np.array(encode_sentence(x,vocab2index)))

    #Stvaramo oznake i spajamo ih sa pripadajucim podatcima
    labels = np.ones(len(df['benefits_data'])).astype(np.int)*0
    zipped = list(zip(df['benefits_data'],labels))

    labels = np.ones(len(df['info_data'])).astype(np.int)
    zipped = zipped + list(zip(df['info_data'],labels))

    labels = np.ones(len(df['requirements_data'])).astype(np.int)*2
    zipped = zipped + list(zip(df['requirements_data'],labels))

    #Uklanjamo prazne stringove
    zipped = [pair for pair in zipped if int(np.sum(pair[0][0])) > 0]
    X, y = [list(c) for c in zip(*zipped)]
    del zipped

    #Podjela na train i test podatke, 10% za test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=48)

    #Klasa kojom dohvacamo dataset
    valid_ds = TextDataset(X_test, y_test)
    train_ds = TextDataset(X_train, y_train)

    #Metaparametri modela
    default_config = dict(
        classes=3,
        embedding_dim=200,
        hidden_dim=[32,32,32],
        batch_size=32,
        learning_rate=0.0001,
        epochs = 140,
        dropout=0.2,
        dataset="private")

    #Pokretanje treniranja
    model = model_pipeline(words, train_ds, valid_ds, default_config)
    torch.save(model, "model_3layers_32_32_32.pt")



    





