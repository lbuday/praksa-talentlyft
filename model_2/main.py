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
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          BertConfig,
                          BertTokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          BertForSequenceClassification)
from utils import BertClassificationCollator
from eval import validation
from data import check_text, tokenize, JobTypeDataset
from train import train


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

    #Stvaramo oznake i spajamo ih sa pripadajucim podatcima
    labels = np.ones(len(df['benefits_data'])).astype(np.int)*0
    zipped = list(zip(df['benefits_data'],labels))

    labels = np.ones(len(df['info_data'])).astype(np.int)
    zipped = zipped + list(zip(df['info_data'],labels))

    labels = np.ones(len(df['requirements_data'])).astype(np.int)*2
    zipped = zipped + list(zip(df['requirements_data'],labels))

    #Uklanjamo prazne stringove
    zipped = [pair for pair in zipped if len(pair[0]) > 0]
    X, y = [list(c) for c in zip(*zipped)]
    del zipped
    X = [ " ".join(x) for x in X ]

    #Podjela na train i test podatke, 10% za test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=48)

    #metaparametri za bert model
    set_seed(123)
    epochs = 4
    batch_size = 32
    max_length = 60
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = 'bert-base-uncased'
    labels_ids = {0: 0, 1: 1, 2: 2}
    n_labels = 3
    
    model_config = BertConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

    #Stvaranje tokenizatora i dodavanje posebnih tokena
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    tokenizer.add_tokens(['NUM'])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #tokenizer.padding_side = "left"
    #tokenizer.pad_token = tokenizer.eos_token

    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    model.to(device)
    bert_classificaiton_collator = BertClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)

    train_dataset = JobTypeDataset(X_train, y_train, use_tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=bert_classificaiton_collator)

    valid_dataset =  JobTypeDataset(X_test, y_test, use_tokenizer=tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=bert_classificaiton_collator)

    optimizer = AdamW(model.parameters(), lr = 2e-5)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    for epoch in tqdm(range(epochs)):
      print()
      print('Training on batches...')
      train_labels, train_predict, train_loss = train(train_dataloader, device, model, optimizer, scheduler)
      train_acc = accuracy_score(train_labels, train_predict)

      print('Validation on batches...')
      valid_labels, valid_predict, val_loss = validation(valid_dataloader, device, model)
      val_acc = accuracy_score(valid_labels, valid_predict)

      print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
      print()

      all_loss['train_loss'].append(train_loss)
      all_loss['val_loss'].append(val_loss)
      all_acc['train_acc'].append(train_acc)
      all_acc['val_acc'].append(val_acc)

    torch.save(model, "model_bert.pt")