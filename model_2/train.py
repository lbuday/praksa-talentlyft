import torch
from tqdm import tqdm

def train(dataloader, device_, model, optimizer, scheduler):
  """Trenira model sa jednim prolayom kroy DataLoader

  Arguments:

    dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
        Podatci parsirani u batcheve tenzora.

    device_ (:obj:`torch.device`):
        Uredaj na kojem treniramo model.

    optimizer (:obj:`transformers.optimization.AdamW`):
        Optimizator koji koristimo tokom treninga

    scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`):
        PyTorch scheduler.

  Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss].
  """
  predictions_labels = []
  true_labels = []
  total_loss = 0

  model.train()

  for batch in tqdm(dataloader, total=len(dataloader)):

    true_labels += batch['labels'].numpy().flatten().tolist()
    
    #Prebacimo sve elemente u batchu na uredaj
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

    model.zero_grad()

    outputs = model(**batch)

    loss, logits = outputs[:2]
    total_loss += loss.item()
    loss.backward()

    #podrezivanje gradijenta iznad 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    logits = logits.detach().cpu().numpy()
    predictions_labels += logits.argmax(axis=-1).flatten().tolist()

  avg_epoch_loss = total_loss / len(dataloader)
  return true_labels, predictions_labels, avg_epoch_loss