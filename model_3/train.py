from utils import get_special_tokens, flat_accuracy, annot_confusion_matrix
import torch
import numpy as np
from tqdm import trange
from sklearn.metrics import f1_score

def train_and_save_model(
    model,
    tokenizer,
    optimizer,
    idx2tag,
    tag2idx,
    this_run,
    max_grad_norm,
    device,
    train_dataloader,
    valid_dataloader,
):
    """Trenira i sprema model"""

    print("start of training")
    pad_tok, sep_tok, cls_tok, o_lab = get_special_tokens(tokenizer, tag2idx)
    verbose = True
    epochs = 12

    epoch = 0
    prev_main_loss, prev_main_loss_counter = 200., 0
    for _ in trange(epochs, desc="Epoch"):
        epoch += 1

        # Training loop
        print("Starting training loop.")
        model.train()
        tr_loss, tr_accuracy, tr_f1 = 0, 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []

        for step, batch in enumerate(train_dataloader):

            # Add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                labels=b_labels,
            )

            loss, tr_logits = outputs[:2]

            # Backward pass
            loss.backward()

            # Compute train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            # Subset out unwanted predictions on CLS/PAD/SEP tokens
            preds_mask = (
                (b_input_ids != cls_tok)
                & (b_input_ids != pad_tok)
                & (b_input_ids != sep_tok)
            )

            tr_logits = tr_logits.detach().cpu().numpy()
            tr_label_ids = torch.masked_select(b_labels, (preds_mask == 1))
            #provjerit len i ako je samo jedan elem u batchu  onda ne squeezat
            try:
              temp = tr_logits[preds_mask.squeeze().cpu()]
            except:
              temp = tr_logits[preds_mask.cpu()]
            tr_batch_preds = np.argmax(temp, axis=1)
            tr_batch_labels = tr_label_ids.detach().cpu().numpy()
            tr_preds.extend(tr_batch_preds)
            tr_labels.extend(tr_batch_labels)


            # Compute training accuracy
            tmp_tr_accuracy = flat_accuracy(tr_batch_labels, tr_batch_preds)
            tr_accuracy += tmp_tr_accuracy

            try:
                tmp_tr_f1 = f1_score(tr_batch_labels, tr_batch_preds, average='micro')
                tr_f1 += tmp_tr_f1
            except:
                print(tr_batch_labels.shape, tr_batch_preds.shape)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
            )

            # Update parameters
            optimizer.step()
            model.zero_grad()

        tr_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        tr_f1 = tr_f1 / nb_tr_steps

        # Print training loss and accuracy per epoch
        print(f"Train loss: {tr_loss}")
        print(f"Train f1 score: {tr_f1}")
        print(f"Train accuracy: {tr_accuracy}")

        # Validation loop
        print("Starting validation loop.")

        model.eval()
        eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []

        for batch in valid_dataloader:

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    labels=b_labels,
                )
                tmp_eval_loss, logits = outputs[:2]

            # Subset out unwanted predictions on CLS/PAD/SEP tokens
            preds_mask = (
                (b_input_ids != cls_tok)
                & (b_input_ids != pad_tok)
                & (b_input_ids != sep_tok)
            )

            logits = logits.detach().cpu().numpy()
            label_ids = torch.masked_select(b_labels, (preds_mask == 1))
            val_batch_preds = np.argmax(logits[preds_mask.squeeze().cpu()], axis=1)
            val_batch_labels = label_ids.cpu().numpy()
            predictions.extend(val_batch_preds)
            true_labels.extend(val_batch_labels)

            tmp_eval_accuracy = flat_accuracy(val_batch_labels, val_batch_preds)
            try:
                tmp_eval_f1 = f1_score(val_batch_labels, val_batch_preds, average='micro')
                eval_f1 += tmp_eval_f1
            except:
                print(val_batch_labels.shape, val_batch_preds.shape)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        # Evaluate loss, acc, conf. matrix, and class. report on devset
        pred_tags = [idx2tag[i] for i in predictions]
        valid_tags = [idx2tag[i] for i in true_labels]
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        eval_f1 = eval_f1 / nb_eval_steps
        conf_mat = annot_confusion_matrix(valid_tags, pred_tags)

        # Report metrics
        print(f"Validation loss: {eval_loss}")
        print(f"Validation F1 score: {eval_f1}")
        print(f"Validation Accuracy: {eval_accuracy}")
        if verbose:
            print(f"Confusion Matrix:\n {conf_mat}")

        if prev_main_loss < eval_loss:
            prev_main_loss_counter += 1
        else:
            prev_main_loss_counter = 0

        prev_main_loss = eval_loss

        if prev_main_loss_counter == 2:
            print("DONE")
            return