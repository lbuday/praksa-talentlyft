import torch

def validation_metrics (model, valid_dl, loss_func, device, accuracy_metric, y_to_device):
  """Funkcija koja tvalidira model

  Arguments:
    model (:obj):
      Model koji zelimo trenirat.
    valid_dl (`DataLoader`):
      Data loader za validacijske podatke.
    loss_func (:obj):
      Loss funkcija kojom treniramo.
    device (:obj):
      Uredaj na kojem treniramo.
    accuracy_metric (:func):
      Funkcija kojom provjeravamo preciznost.
    y_to_device (:func):
      Funkcija kojom prebacujemo y na uredaj.

  Returns:
    `float`: Loss validacijskih podataka,
    `float`: Preciznost validacijskih podataka
    """
  model.eval()
  accuracy = 0
  count = 0
  total = 0
  sum_loss = 0.0
  for x, y, l in valid_dl:
    x = x.long().to(device)
    y = y_to_device(y, device)
    y_hat = model(x, l)
    loss = loss_func(y_hat, y)
    #prima posebnu funkciju za preciznost jer ona ovisi o modelu
    accuracy += accuracy_metric(y_hat, y)
    total += x.shape[0]
    count += 1
    sum_loss += loss.item()*x.shape[0]
  return sum_loss/total, accuracy/total