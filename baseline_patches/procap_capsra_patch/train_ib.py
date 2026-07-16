import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup

from dataset import Multimodal_Data
import utils


def bce_for_loss(logits, labels):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_auc_score(logits, label):
    batch_size = logits.shape[0]
    auc = roc_auc_score(label.cpu().numpy(), logits.cpu().numpy(), average="weighted") * batch_size
    return auc


def compute_score(logits, labels):
    preds = torch.max(logits, 1)[1]
    one_hot = torch.zeros(*labels.size()).cuda()
    one_hot.scatter_(1, preds.view(-1, 1), 1)
    return (one_hot * labels).sum().float()


def compute_scaler_score(logits, labels):
    logits = torch.max(logits, 1)[1]
    labels = labels.squeeze(-1)
    return (logits == labels).int().sum().float()


def train_for_epoch(opt, model, train_loader, test_loader):
    model.set_split_data(opt.train_neighbor_path, opt.train_feature_path)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large") if opt.MODEL == "pbm" and opt.USE_DEMO and opt.MULTI_QUERY else None

    log_path = os.path.join(opt.DATASET)
    os.makedirs(log_path, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = utils.Logger(os.path.join(log_path, f"{opt.SAVE_NUM}_seed{opt.SEED}+GAT_IB_{current_time}.txt"))
    logger.write("=== Hyperparameters ===")
    for k, v in vars(opt).items():
        logger.write(f"{k} : {v}")
    logger.write(f"Length of training set: {len(train_loader.dataset)}, test set: {len(test_loader.dataset)}")
    logger.write(f"Max input length: {model.max_length}")
    logger.write(f"Initial IB beta: {getattr(model, 'ib_beta', 'N/A')}")
    logger.write("=======================")

    roberta_params, gat_params, cls_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("roberta"):
            roberta_params.append(p)
        elif "gat" in n:
            gat_params.append(p)
        else:
            cls_params.append(p)

    optimizer = AdamW(
        [
            {"params": roberta_params, "lr": opt.LR_RATE},
            {"params": gat_params, "lr": opt.GAT_LR},
            {"params": cls_params, "lr": opt.CLS_LR},
        ],
        eps=opt.EPS,
    )
    total_steps = len(train_loader) * opt.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    record_auc = []
    record_acc = []

    for epoch in range(opt.EPOCHS):
        model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_kl_loss = 0.0
        running_score = 0.0

        for batch in train_loader:
            targets = batch["target"].cuda()
            raw_ids = batch["img"]
            meme_ids = [os.path.splitext(x)[0] for x in raw_ids]
            text = batch["prompt_all_text"] if opt.USE_DEMO else batch["test_all_text"]

            fused_logits, kl_loss = model(meme_ids, text)
            main_loss = bce_for_loss(fused_logits, targets)
            ib_beta = getattr(model, "ib_beta", 0.0) or 0.0
            total_loss = main_loss + ib_beta * kl_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            score = compute_score(fused_logits, targets)
            running_loss += total_loss.item()
            running_main_loss += main_loss.item()
            running_kl_loss += kl_loss.item()
            running_score += score

        model.set_split_data(opt.test_neighbor_path, opt.test_feature_path)
        model.eval()
        if opt.MODEL == "pbm" and opt.USE_DEMO and opt.MULTI_QUERY:
            eval_acc, eval_auc = eval_multi_model(opt, model, tokenizer)
        else:
            eval_acc, eval_auc = eval_model(opt, model, test_loader)

        record_acc.append(eval_acc)
        record_auc.append(eval_auc)
        avg_train_acc = (running_score / len(train_loader.dataset)) * 100
        avg_epoch_main = running_main_loss / len(train_loader)
        avg_epoch_kl = running_kl_loss / len(train_loader)
        avg_epoch_tot = running_loss / len(train_loader)

        logger.write(f"--- Epoch {epoch} Summary ---")
        logger.write(
            f"Train: Avg Main Loss={avg_epoch_main:.6f}, Avg KL Loss={avg_epoch_kl:.6f}, Avg Total Loss={avg_epoch_tot:.6f}, Train Acc={avg_train_acc:.2f}%"
        )
        logger.write(f"Test:  AUC={eval_auc:.2f}, Acc={eval_acc:.2f}")
        logger.write(f"Current IB beta: {ib_beta:.6f}")
        logger.write("==============================")
        model.set_split_data(opt.train_neighbor_path, opt.train_feature_path)

    best = max(range(len(record_auc)), key=lambda i: record_auc[i] + record_acc[i])
    logger.write(f"Best Epoch {best}: AUC={record_auc[best]:.2f}, Acc={record_acc[best]:.2f}")


def eval_model(opt, model, test_loader):
    total_score = 0.0
    all_logits = []
    all_labels = []

    for batch in test_loader:
        with torch.no_grad():
            labels = batch["label"].float().cuda().view(-1, 1)
            targets = batch["target"].cuda()
            raw_ids = batch["img"]
            meme_ids = [os.path.splitext(x)[0] for x in raw_ids]
            text = batch["prompt_all_text"] if opt.USE_DEMO else batch["test_all_text"]
            fused_logits, _ = model(meme_ids, text)
            total_score += compute_score(fused_logits, targets)
            probs = F.softmax(fused_logits, dim=-1)[:, 1].unsqueeze(-1)
            all_logits.append(probs)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    auc = compute_auc_score(all_logits, all_labels)
    acc = total_score * 100.0 / len(test_loader.dataset)
    return acc, auc


def eval_multi_model(opt, model, tokenizer):
    num_queries = opt.NUM_QUERIES
    labels_record = {}
    logits_record = {}
    prob_record = {}
    len_data = None
    for k in range(num_queries):
        test_set = Multimodal_Data(opt, opt.DATASET, "test")
        test_loader = torch.utils.data.DataLoader(test_set, opt.BATCH_SIZE, shuffle=False, num_workers=2)
        len_data = len(test_loader.dataset)
        for batch in test_loader:
            with torch.no_grad():
                label = batch["label"].float().cuda().view(-1, 1)
                img = batch["img"]
                raw_ids = batch["img"]
                meme_ids = [os.path.splitext(x)[0] for x in raw_ids]
                text = batch["prompt_all_text"]
                fused_logits, _ = model(meme_ids, text)
                norm_prob = F.softmax(fused_logits, dim=-1)
                norm_logits = norm_prob[:, 1].unsqueeze(-1)
                for j in range(norm_prob.shape[0]):
                    cur_img = img[j]
                    if k == 0:
                        labels_record[cur_img] = label[j : j + 1]
                        logits_record[cur_img] = norm_logits[j : j + 1]
                        prob_record[cur_img] = norm_prob[j : j + 1]
                    else:
                        logits_record[cur_img] += norm_logits[j : j + 1]
                        prob_record[cur_img] += norm_prob[j : j + 1]

    labels, logits, probs = [], [], []
    for name in labels_record.keys():
        labels.append(labels_record[name])
        logits.append(logits_record[name] / num_queries)
        probs.append(prob_record[name] / num_queries)

    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    scores = compute_scaler_score(probs, labels)
    auc = compute_auc_score(logits, labels)
    return scores * 100.0 / len_data, auc * 100.0 / len_data