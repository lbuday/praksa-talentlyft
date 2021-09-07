import torch
from tqdm import tqdm

def validation(dataloader, device_, model):
    """Funkcija koja evaluira performansu modela na odvojenim podatcima  

    Arguments:  
      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsirani podatci u batcheve tenzora    
      device_ (:obj:`torch.device`):
            Uredaj na kojem radimo evaluaciju.   

    Returns:
      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss]
    """
    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):

      true_labels += batch['labels'].numpy().flatten().tolist()

      batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

      with torch.no_grad():        
          outputs = model(**batch)

          loss, logits = outputs[:2]
          logits = logits.detach().cpu().numpy()
          total_loss += loss.item()

          predict_content = logits.argmax(axis=-1).flatten().tolist()
          predictions_labels += predict_content

    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_epoch_loss