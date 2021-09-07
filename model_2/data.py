from torch.utils.data import Dataset, DataLoader
import re
from langdetect import detect
import pandas as pd

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

class JobTypeDataset(Dataset):
  def __init__(self, X, y, use_tokenizer):
    self.texts = X
    self.labels = y
    self.n_examples = len(self.labels)
    return None

  def __len__(self):
    return self.n_examples

  def __getitem__(self, item):
    return {'text':self.texts[item],
            'label':self.labels[item]}