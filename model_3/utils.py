import argparse
import datetime as dt
import numpy as np
import os
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
import re
from sklearn.metrics import confusion_matrix
import spacy
import subprocess
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification, AutoConfig

def annot_confusion_matrix(valid_tags, pred_tags):

    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    # Create header from unique tags
    header = sorted(list(set(valid_tags + pred_tags)))

    # Calculate the actual confusion matrix
    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    # Final formatting touches for the string output
    mat_formatted = [header[i] + "\t" + str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content

def flat_accuracy(valid_tags, pred_tags):
    """Preciznost koju koristimo tokom treniranja
    """
    try:
      out = (np.array(valid_tags) == np.array(pred_tags)).mean()
    except:
      #u slucaju da je batch velicine 1 vraca vrijenost a ne bool
      if (np.array(valid_tags) == np.array(pred_tags)):
        out = 1
      else:
        out = 0
    return out

def get_special_tokens(tokenizer, tag2idx):

    pad_tok = tokenizer.vocab["[PAD]"]
    sep_tok = tokenizer.vocab["[SEP]"]
    cls_tok = tokenizer.vocab["[CLS]"]
    o_lab = tag2idx["O"]

    return pad_tok, sep_tok, cls_tok, o_lab

def get_hyperparameters(model, ff):

    # ff: full_finetuning
    if ff:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters

