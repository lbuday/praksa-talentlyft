import pandas as pd
import numpy as np
import re
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader
import torch

def check_text(text: str, lang: str='en') -> bool:
  """Vraca bool vrijednost ovisno o jeziku teksta

  Arguments:
    text (`str`):
      Recenica u obliku stringa.
    lang (`str`):
      Jezik koji zelimo prepoznat.

  Returns:
    `bool`: Bool vrijednost, True ako pripada jeziku.
  """
  #filtriramo nan vrijednosti
  if pd.isna(text):
    return False
  else:
    cleanr = re.compile('<.*?>')
    clean_text = re.sub(cleanr, '', text)
    try:
      return detect(clean_text) == lang
    except:
      return False


def tokenize(text: str, en_stops):
  """Podjeli string u listu podstringova

  Arguments:
    text (`str`):
      Recenica u obliku stringa.
    en_stops (`str`):
      Stop rijeci engleskog jezika.

  Returns:
    `List[str]`: Lista tokena u obliku stringova.
  """
  cleanr = re.compile('<.*?>')

  #Pretvara sve u mala slova i uklanja html kod
  nopunct = re.sub(cleanr, '', text.lower())
  #Sve brojeve zamjeni sa tokenom NUM
  nopunct = re.sub( r"[0-9]+"," NUM ", nopunct)
  #Uklanja sve sto nije slovo
  nopunct = re.sub( r"[^a-zA-Z]+"," ", nopunct)
  #Uklanja dodatne razmake
  nopunct = re.sub( r" +"," ", nopunct)
  nopunct = nopunct.strip()

  #Uklanja stop rijeci
  new_words = []
  for word in nopunct.split(): 
    if word not in en_stops:
        new_words.append(word)
  return new_words


def encode_sentence(text, vocab2index, N=1024):
  """Dodaje padding i pretvara rijeci u njihove indekse

  Arguments:
    text (`List[str]`):
      Ranije tokenizirana recenica.
    vocab2index (`Dict[str,int]`):
      Rijecnik indeksa.
    N (`int`) = 1024:
      Maksimalna duljina recenice

  Returns:
    `List[int]`: Lista indeksa,
    `int`: Duljina ne paddanog dijela
  """
  encoded = np.zeros(N, dtype=int)
  #Uklanja prazne vrijednosti
  if text == None or text == []:
    return encoded, 0
  tokenized = text
  #Dodaje START i END token na pocetak i kraj
  tokenized = tokenized[:N-2]
  tokenized = ["START"] + tokenized + ["END"]
  #Zamjenjuje rijeci sa indeksima
  enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])

  #Uzima maksimalnu duljinu i radi padding
  length = min(N-1, len(enc1))
  encoded[:length] = enc1[:length]
  return encoded, length


class TextDataset(Dataset):
  """Klasa koja dohvaca elemente sa odredenog indeksa"""
  def __init__(self, X, Y):
    self.X = X
    self.y = Y
      
  def __len__(self):
    #Kad koristimo len vraca broj primjera
    return len(self.y)
  
  def __getitem__(self, idx):
    #Vraca X, y i l koji je duljina od trenutnog X (bez paddinga)
    return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]