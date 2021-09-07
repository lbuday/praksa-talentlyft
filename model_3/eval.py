import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import spacy
import numpy as np
import pandas as pd
import pickle
import re
from spacy.tokens import Doc
from spacy import displacy
from spacy.vocab import Vocab

def load_model_and_tokenizer(model_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('model_bert_NER.pt')
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    tokenizer.add_tokens(['[NUMBER]'])
    tokenizer.add_tokens(['[MONEY]'])
    tokenizer.add_tokens(['[LINK]'])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return model, tokenizer

def get_bert_pred_df(model, tokenizer, input_text, label_dict):
    """
    Uses the model to make a prediction, with batch size 1.
    """
    new_elem = []
    for e in input_text.split():
      if re.match(r'[\$£€¤][0-9].+', e):
        new_elem.append('<MONEY>')
      elif re.match(r'[0-9].+', e):
        new_elem.append('<NUMBER>')
      elif re.match(r'www\..+', e):
        new_elem.append('<LINK>')
      else:
        new_elem.append(e)
    
    vocab = Vocab(strings=new_elem)
    input_text = " ".join(new_elem)

    encoded_text = tokenizer.encode(input_text)
    wordpieces = [tokenizer.decode(tok).replace(" ", "") for tok in encoded_text]

    print("here0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = torch.tensor(encoded_text).unsqueeze(0).long().to(device)
    labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0).long().to(device)
    print("here1")
    outputs = model(input_ids, labels=labels)
    print("here2")
    loss, scores = outputs[:2]
    scores = scores.detach().cpu().numpy()
    

    label_ids = np.argmax(scores, axis=2)
    preds = [label_dict[i] for i in label_ids[0]]

    wp_preds = list(zip(wordpieces, preds))
    toplevel_preds = [pair[1] for pair in wp_preds if "##" not in pair[0]]
    str_rep = " ".join([t[0] for t in wp_preds]).replace(" ##", "").split()

    # filling the pandas dataset
    if len(str_rep) == len(toplevel_preds):
        preds_final = list(zip(str_rep, toplevel_preds))
        b_preds_df = pd.DataFrame(preds_final)
        b_preds_df.columns = ["text", "pred"]
        for tag in ["A", "M", "R"]:
            b_preds_df[f"b_pred_{tag.lower()}"] = np.where(
                b_preds_df["pred"].str.contains("B-"+tag) | b_preds_df["pred"].str.contains("I-"+tag), 1, 0
            )

            b_preds_df[f"pred_no_bio"] = np.where(
                b_preds_df["pred"].str.contains("B-"+tag) | b_preds_df["pred"].str.contains("I-"+tag), tag, "O"
            )
        return b_preds_df.loc[:, "text":], vocab
    else:
        print("Could not match up output string with preds.")
        return None

if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer('pucpr/clinicalnerpt-quantitative')

    label_types = ["B-A", "I-A", "B-M", "I-M", "B-R", "I-R", "O"]
    tag2idx = {t: i for i, t in enumerate(label_types)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    input_text = " ".join((pickle.load( open( "../sent_tokens.p", "rb" ) ))[0][0:400])

    df, vocab = get_bert_pred_df(
        model, tokenizer, input_text, idx2tag
    )

    doc = Doc(vocab, words=list(df.text), tags=list(df.pred_no_bio))
    pickle.dump( doc, open( "doc.p", "wb" ) )
    #displacy.serve(doc, style="ent")
