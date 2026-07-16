import datetime
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import utils


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            raw_ids = batch["img"]
            meme_ids = [os.path.splitext(x)[0] for x in raw_ids]
            tokens = batch["cap_tokens"].long().to(device)
            mask = batch["mask"].to(device)
            mask_pos = batch.get("mask_pos", None)
            if mask_pos is not None:
                mask_pos = mask_pos.to(device)

            labels_idx = batch["label"].long().to(device)
            labels = torch.zeros(labels_idx.size(0), num_classes, device=device)
            labels.scatter_(1, labels_idx.unsqueeze(1), 1.0)

            logits = model(meme_ids, tokens, mask, mask_pos)
            probs = torch.sigmoid(logits)
            all_logits.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    try:
        auc = roc_auc_score(all_labels, all_logits, average="weighted")
    except ValueError:
        auc = 0.0
    pred_labels = (all_logits >= 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels, average="weighted")
    return auc, acc, f1


def train_for_epoch(opt, model, train_loader, dev_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if opt.SAVE:
        model_dir = os.path.join(opt.SAVE_DIR, f"{opt.MODEL}_{opt.DATASET}")
        os.makedirs(model_dir, exist_ok=True)

    log_dir = getattr(opt, "LOG_DIR", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = utils.Logger(os.path.join(log_dir, f"{opt.SAVE_NUM}_{now}.txt"))

    for k, v in vars(opt).items():
        logger.write(f"{k}: {v}")
    logger.write(f"Train set size: {len(train_loader.dataset)}, Dev set size: {len(dev_loader.dataset)}")

    roberta_params, gat_params, gat_clf_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "roberta" in name:
            roberta_params.append(param)
        elif "gat." in name:
            gat_params.append(param)
        elif "gat_classifier" in name:
            gat_clf_params.append(param)
        else:
            roberta_params.append(param)

    optimizer = AdamW(
        [
            {"params": roberta_params, "lr": opt.LR_RATE, "weight_decay": opt.WEIGHT_DECAY},
            {"params": gat_params, "lr": opt.lr_gat, "weight_decay": opt.WEIGHT_DECAY},
            {"params": gat_clf_params, "lr": opt.lr_clf, "weight_decay": opt.WEIGHT_DECAY},
        ],
        lr=opt.LR_RATE,
        eps=opt.EPS,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=getattr(opt, "NUM_WARMUP_STEPS", 0),
        num_training_steps=len(train_loader) * opt.EPOCHS,
    )

    best_metric = -float("inf")
    num_classes = len(model.label_word_list)

    for epoch in range(opt.EPOCHS):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            raw_ids = batch["img"]
            meme_ids = [os.path.splitext(x)[0] for x in raw_ids]
            tokens = batch["cap_tokens"].long().to(device)
            mask = batch["mask"].to(device)
            mask_pos = batch.get("mask_pos", None)
            if mask_pos is not None:
                mask_pos = mask_pos.to(device)

            labels_idx = batch["label"].long().to(device)
            labels = torch.zeros(labels_idx.size(0), num_classes, device=device)
            labels.scatter_(1, labels_idx.unsqueeze(1), 1.0)

            optimizer.zero_grad()
            logits = model(meme_ids, tokens, mask, mask_pos)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        logger.write(f"Epoch {epoch + 1} train loss: {avg_loss:.4f}")

        val_auc, val_acc, val_f1 = evaluate(model, dev_loader, device, num_classes)
        logger.write(f"Epoch {epoch + 1} val AUC: {val_auc:.4f}, ACC: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_auc > best_metric:
            best_metric = val_auc
            if opt.SAVE:
                save_path = os.path.join(opt.SAVE_DIR, f"{opt.MODEL}_{opt.DATASET}", "best_model.pth")
                torch.save(model.state_dict(), save_path)
                logger.write(f"Best model saved at epoch {epoch + 1} with AUC {val_auc:.4f}")