from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class BertDataset:
    """Rijesava tokenizaciju i pretvara oznake u indekse

    Arguments:
        data (`Dict[str, list[str]]`):
          Podatci koje koristimo podjeljeni na train i test,
          koji su podjeljeni na X i y.
        tokenizer (:transformers.tokenizer):
          Tokenizator koji koristimo.
        max_len (`int`):
          maksimalna duljina ulaza.
        tag2idx (`Dict[str,int]`):
          mapiranje iz oznake u indeks.
    """
    def __init__(self, data, tokenizer, max_len, tag2idx):

        pad_tok = tokenizer.vocab["[PAD]"]
        sep_tok = tokenizer.vocab["[SEP]"]
        o_lab = tag2idx["O"]

        #Tokeniziraj tekst u podrijeci tako da odrzis oznake
        tokenized_texts = [
            tokenize_and_preserve_labels(sent, labs, tokenizer)
            for sent, labs in zip(data["sents"], data["labels"])
        ]

        self.toks = [["[CLS]"] + text[0] for text in tokenized_texts]
        self.labs = [["O"] + text[1] for text in tokenized_texts]

        #Pretvori tokene u indekse
        self.input_ids = pad_sequences(
            [tokenizer.convert_tokens_to_ids(txt) for txt in self.toks],
            maxlen=max_len,
            dtype="long",
            truncating="post",
            padding="post",
        )

        #Pretvori oznake u indekse
        self.tags = pad_sequences(
            [[tag2idx.get(l) for l in lab] for lab in self.labs],
            maxlen=max_len,
            value=tag2idx["O"],
            padding="post",
            dtype="long",
            truncating="post",
        )

        # Zamjenjuje posljednji token-label par sa ([SEP], O)
        # za bilo koju listu koja dosegne MAX_LEN
        for voc_ids, tag_ids in zip(self.input_ids, self.tags):
            if voc_ids[-1] == pad_tok:
                continue
            else:
                voc_ids[-1] = sep_tok
                tag_ids[-1] = o_lab

        self.attn_masks = [[float(i > 0) for i in ii] for ii in self.input_ids]


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """Word piece tokenizacija otezava spajanje rijeci sa njihovim oznakama,
    ovom funkcijom odzavamo oznake sa pripadajucim rijecima i podrijecima

    Arguments:
        sentence (`List[str]`):
          recenica koju tokeniziramo.
        text_labels (`List[str]`):
          oznake na danoj recenici.
        tokenizer (:transformers.tokenizer):
          Tokenizator koji koristimo.

    Returns:
        `List[int]`: tokenizirana recenica,
        `List[str]`: oznake prosirene na podrijeci.
    """
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokeniziraj rijec i izbroji u koliko podrijeci je podjeljena
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Dodaj tokeniziranu rijec u konacnu listu
        tokenized_sentence.extend(tokenized_word)

        #Produlji oznake ovisno o duljini podrijeci u rijeci
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def load_and_prepare_data(data, tokenizer, max_len, batch_size, tag2idx):
    """Pravi DataLoadere od podataka

    Arguments:
        data (`Dict[str, list[str]]`):
          Podatci koje koristimo podjeljeni na train i test,
          koji su podjeljeni na X i y.
        tokenizer (:transformers.tokenizer):
          Tokenizator koji koristimo.
        max_len (`int`):
          maksimalna duljina ulaza.
        batch_size (`int`):
          velicina batcha.
        tag2idx (`Dict[str,int]`):
          mapiranje iz oznake u indeks.

    Returns:
        :obj:`torch.utils.data.dataloader.DataLoader`: dataloader za trening podatke,
        :obj:`torch.utils.data.dataloader.DataLoader`: dataloader za test podatke.
    """

    train = BertDataset(data["train"], tokenizer, max_len, tag2idx)
    dev = BertDataset(data["test"], tokenizer, max_len, tag2idx)

    tr_inputs = torch.tensor(train.input_ids)
    val_inputs = torch.tensor(dev.input_ids)
    tr_tags = torch.tensor(train.tags)
    val_tags = torch.tensor(dev.tags)
    tr_masks = torch.tensor(train.attn_masks)
    val_masks = torch.tensor(dev.attn_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(
        valid_data, sampler=valid_sampler, batch_size=batch_size
    )

    return train_dataloader, valid_dataloader