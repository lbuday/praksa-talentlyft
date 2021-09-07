import wandb
import torch
from utils import train_log
from eval import validation_metrics

def train_model(model, train_dl, val_dl, device, criterion, accuracy_metric, y_to_device, epochs=10, lr=0.1, print_every=1):
  """Funkcija koja trenira model

  Arguments:
    model (:obj):
      Model koji zelimo trenirat.
    train_dl, valid_dl (`DataLoader`):
      Data loader za trening i validacijske podatke.
    device (:obj):
      Uredaj na kojem treniramo.
    criterion (:obj):
      Loss funkcija kojom treniramo.
    accuracy_metric (:func):
      Funkcija kojom provjeravamo preciznost.
    y_to_device (:func):
      Funkcija kojom prebacujemo y na uredaj.
    epochs (`int`)=10:
      Broj epocha.
    lr (`float`)=0.1:
      Stopa ucenja.
    print_every (`int`)=1:
      Nakon kojeg epocha radimo ispis

  Returns: :torch.model
  """
  loss_func = criterion
  wandb.watch(model)

  #Salje optimizatoru samo parametre koje moze mjenjat
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = torch.optim.Adam(parameters,lr=lr)
  
  prev_main_loss = 0
  prev_main_loss_counter = 0

  for i in range(epochs):
    model.train()
    sum_loss = 0.0
    total = 0
    correct = 0
    for x, y, l in train_dl:
      x = x.long().to(device)
      #y prebacujemo na GPU
      y = y_to_device(y, device)
      y_pred = model(x, l)
      #saljemo predikciju zeljenoj loss funkciji
      loss = loss_func(y_pred, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      sum_loss += loss.item()*x.shape[0]
      total += x.shape[0]
    
    #Validacija podataka
    val_loss, val_acc = validation_metrics(model, val_dl, loss_func, device, accuracy_metric, y_to_device)
    if i % print_every == 0:
      train_log(val_loss, total, i)
      print("train loss {0:.3f}, val loss {1:.7f}, val accuracy {2}".format(sum_loss/total, val_loss, val_acc))

    #early stopping code
    if prev_main_loss < val_loss:
      prev_main_loss_counter += 1
    else:
      prev_main_loss_counter = 0

    prev_main_loss = val_loss

    if prev_main_loss_counter == 2:
        print("DONE")
        return model