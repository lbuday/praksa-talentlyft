import argparse
import datetime as dt
import numpy as np
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import spacy
import subprocess
import torch
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import pickle
from utils import get_hyperparameters
from data import load_and_prepare_data
from train import train_and_save_model

if __name__ == "__main__":
    #ucitavanje podataka
    sent_tokens = pickle.load( open( "../sent_tokens.p", "rb" ) )
    sent_labels = pickle.load( open( "../sent_labels.p", "rb" ) )

    #zamjena odredenih stringova sa tokenima
    new_list = []
    for x in sent_tokens:
      new_elem = []
      for e in x:
        if re.match(r'[\$£€¤][0-9].+', e):
          new_elem.append('<MONEY>')
        elif re.match(r'[0-9].+', e):
          new_elem.append('<NUMBER>')
        elif re.match(r'www\..+', e):
          new_elem.append('<LINK>')
        else:
          new_elem.append(e)
      new_list.append(new_elem)

    sent_tokens = new_list
    del new_list

    #skracivanje oznaka
    new_list = []
    for x in sent_labels:
      new_elem = []
      for e in x:
        if e == 'OTHER':
          new_elem.append('O')
        elif e == 'ABOUT COMPANY':
          new_elem.append('A')
        elif e == 'RESPONSIBILITIES':
          new_elem.append('R')
        else:
          new_elem.append('M')
      new_list.append(new_elem)

    sent_labels = new_list
    del new_list

    #Pravljenje BIO sheme
    new_list = []
    for x in sent_labels:
      new_elem = []
      prev = 'O'
      for e in x:
        if prev != e and e != 'O':
          new_elem.append('B-'+e)
        elif prev == e and e != 'O':
          new_elem.append('I-'+e)
        else:
          new_elem.append(e)
        prev = e
      new_list.append(new_elem)

    sent_labels = new_list
    del new_list

    #Podjela podataka na train i test
    data = { "train" : {"sents" : None, "labels": None}, "test" : {"sents" : None, "labels": None}}
    data["train"]["sents"], data["test"]["sents"], data["train"]["labels"], data["test"]["labels"] = \
      train_test_split(sent_tokens, sent_labels, test_size=0.10, random_state=48)

    #metaparametri za treniranje
    label_types = ["B-A", "I-A", "B-M", "I-M", "B-R", "I-R", "O"]
    MAX_LEN = 512
    BATCH_SIZE = 2
    EPOCHS = 8
    MODEL = 'pucpr/clinicalnerpt-quantitative'
    THIS_RUN = dt.datetime.now().strftime("%m.%d.%Y, %H.%M.%S")
    MAX_GRAD_NORM = 1.0
    NUM_LABELS = len(label_types)
    FULL_FINETUNING = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Stvaranje tokenizatora i dodavanje posebnih tokena
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_special_tokens=True , do_lower_case=True)
    tokenizer.add_tokens(['[NUMBER]'])
    tokenizer.add_tokens(['[MONEY]'])
    tokenizer.add_tokens(['[LINK]'])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tag2idx = {t: i for i, t in enumerate(label_types)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    train_dataloader, valid_dataloader = load_and_prepare_data(
        data, tokenizer, MAX_LEN, BATCH_SIZE, tag2idx
    )
    print("Loaded training and validation data into DataLoaders.")

    # Initialize model
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=MODEL, num_labels=NUM_LABELS, max_position_embeddings=MAX_LEN)
    bert = AutoModel.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_config(model_config)
    model.bert = bert
    model.to(device)
    print(f"Initialized model and moved it to {device}.")

    optimizer_grouped_parameters = get_hyperparameters(model, FULL_FINETUNING)
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    print("Initialized optimizer and set hyperparameters.")
    train_and_save_model(
        model,
        tokenizer,
        optimizer,
        idx2tag,
        tag2idx,
        THIS_RUN,
        MAX_GRAD_NORM,
        device,
        train_dataloader,
        valid_dataloader,
    )

    torch.save(model, "model_bert_NER.pt")