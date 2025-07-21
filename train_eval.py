import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def train_one_epoch(model, dataloader, optimizer, scheduler, device,
                    feature_path_train, neighbor_path_train, 
                    log_every_batches=50):
    model.train()
    model.load_external_data(feature_path_train, neighbor_path_train, device)
    
    total_loss = total_ce = total_ib = 0.0
    correct_predictions = total_samples = 0
    batch_count = 0

    for batch_idx, (meme_ids, inputs, labels) in enumerate(tqdm(dataloader, desc="Training")):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits, ib_loss = model(meme_ids, inputs, labels)
        
        loss_ce = F.cross_entropy(logits, labels)
        loss = loss_ce + ib_loss
        
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_ib += ib_loss.item()
        
        preds = logits.argmax(dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        batch_count += 1

        if (batch_idx + 1) % log_every_batches == 0:
            avg_ce = total_ce / batch_count
            avg_ib = total_ib / batch_count
            acc = correct_predictions / total_samples if total_samples > 0 else 0.0
            print(f"[Batch {batch_idx+1}/{len(dataloader)}] CE Loss: {avg_ce:.6f}, IB Loss: {avg_ib:.6f}, Acc: {acc:.4f}")

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    train_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    avg_ce_loss = total_ce / batch_count if batch_count > 0 else 0.0
    avg_ib_loss = total_ib / batch_count if batch_count > 0 else 0.0

    return avg_loss, train_acc, avg_ce_loss, avg_ib_loss

def evaluate(model, dataloader, device, feature_path_dev, neighbor_path_dev):
    model.eval()
    model.load_external_data(feature_path_dev, neighbor_path_dev, device)
    
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for meme_ids, inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            logits, _ = model(meme_ids, inputs, labels=None) 
            
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    if len(all_labels) == 0:
        print("Warning: Evaluation set is empty or produced no results.")
        return 0.0, 0.0, 0.0
        
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    
    return macro_f1, auc, acc
