import argparse
import json
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from data.dataset import HatefulMemeDataset, collate_fn
from models.layers import (
    AttentionFusion,
    ConcatFusion,
    GATLayer,
    GatedFusion,
    TransformerConvLayer,
    WeightedAverageLayer,
)
from models.backbones import get_default_model_name, get_feature_dim
from models.model import HatefulMemeClassifierIB
from train_eval import evaluate, train_one_epoch
from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the modular CapSRA mainline model."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="MOMENTA-main/HarMeme_V1/Annotations/Harm-C",
        help="Directory containing the split jsonl files and image folder.",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default=None,
        help="Optional explicit path to the training jsonl file.",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default=None,
        help="Optional explicit path to the validation jsonl file.",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default=None,
        help="Optional explicit path to the test jsonl file.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Optional explicit path to the image directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory for checkpoints, logs, and saved config.",
    )
    parser.add_argument(
        "--feature_path_train",
        type=str,
        default=None,
        help="Feature cache containing the neighbor ids used by the training neighbor json.",
    )
    parser.add_argument(
        "--feature_path_val",
        type=str,
        default=None,
        help="Feature cache containing the neighbor ids used by the validation neighbor json.",
    )
    parser.add_argument(
        "--feature_path_test",
        type=str,
        default=None,
        help="Feature cache containing the neighbor ids used by the test neighbor json. Falls back to val.",
    )
    parser.add_argument(
        "--neighbor_path_train",
        type=str,
        default=None,
        help="Neighbor json for the training split.",
    )
    parser.add_argument(
        "--neighbor_path_val",
        type=str,
        default=None,
        help="Neighbor json for the validation split.",
    )
    parser.add_argument(
        "--neighbor_path_test",
        type=str,
        default=None,
        help="Neighbor json for the test split. Falls back to val.",
    )

    parser.add_argument(
        "--aggregation_type",
        type=str,
        default="GAT",
        choices=["GAT", "TransformerConv", "WeightedAverage"],
        help="Neighbor aggregation module.",
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="Gated",
        choices=["Gated", "Concat", "Attention"],
        help="Feature fusion module.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="clip",
        choices=["clip", "vilt"],
        help="Frozen multimodal backbone used by the CapSRA mainline.",
    )
    parser.add_argument(
        "--encoder_model_name",
        type=str,
        default=None,
        help="Optional Hugging Face model name for the selected base model.",
    )
    parser.add_argument(
        "--encoder_dim",
        type=int,
        default=None,
        help="Feature dimension of the selected frozen backbone. Auto-filled when omitted.",
    )
    parser.add_argument(
        "--agg_out_dim",
        type=int,
        default=None,
        help="Output dimension for the aggregation module. Defaults to encoder_dim.",
    )
    parser.add_argument(
        "--fusion_dim",
        type=int,
        default=None,
        help="Output dimension for the fusion module. Defaults to encoder_dim.",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_warmup_steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="auc",
        choices=["auc", "acc", "f1"],
        help="Validation metric used for early stopping and best checkpointing.",
    )

    parser.add_argument("--gat_hidden_dim", type=int, default=256)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--ib_beta", type=float, default=1e-4)
    parser.add_argument("--ib_dim", type=int, default=256)
    parser.add_argument(
        "--use_ib",
        dest="use_ib",
        action="store_true",
        help="Enable the information bottleneck module.",
    )
    parser.add_argument(
        "--no_ib",
        dest="use_ib",
        action="store_false",
        help="Disable the information bottleneck module.",
    )
    parser.set_defaults(use_ib=True)

    args = parser.parse_args()
    args.train_json = args.train_json or os.path.join(args.data_dir, "train_fix.jsonl")
    args.val_json = args.val_json or os.path.join(args.data_dir, "val_fix.jsonl")
    args.test_json = args.test_json or os.path.join(args.data_dir, "test_fix.jsonl")
    args.image_dir = args.image_dir or os.path.join(args.data_dir, "img")
    args.encoder_model_name = args.encoder_model_name or get_default_model_name(args.base_model)
    args.encoder_dim = args.encoder_dim or get_feature_dim(args.base_model)
    args.agg_out_dim = args.agg_out_dim or args.encoder_dim
    args.fusion_dim = args.fusion_dim or args.encoder_dim
    args.feature_path_test = args.feature_path_test or args.feature_path_val
    args.neighbor_path_test = args.neighbor_path_test or args.neighbor_path_val
    args.log_file = os.path.join(args.output_dir, "training_log.txt")
    args.best_model_path = os.path.join(args.output_dir, "best_model.pt")
    args.config_path = os.path.join(args.output_dir, "run_config.json")
    return args


def build_aggregation_module(args):
    if args.aggregation_type == "GAT":
        return GATLayer(
            in_dim=args.encoder_dim,
            hidden_dim=args.gat_hidden_dim,
            out_dim=args.agg_out_dim,
            heads=args.gat_heads,
        )
    if args.aggregation_type == "TransformerConv":
        return TransformerConvLayer(
            in_dim=args.encoder_dim,
            out_dim=args.agg_out_dim,
            n_heads=args.transformer_heads,
        )
    return WeightedAverageLayer(in_dim=args.encoder_dim, out_dim=args.agg_out_dim)


def build_fusion_module(args, agg_out_dim):
    if args.fusion_type == "Gated":
        if not (args.encoder_dim == agg_out_dim == args.fusion_dim):
            raise ValueError(
                "Gated fusion requires encoder_dim, agg_out_dim, and fusion_dim to be equal."
            )
        return GatedFusion(
            dim1=args.encoder_dim,
            dim2=agg_out_dim,
            fusion_dim=args.fusion_dim,
        )
    if args.fusion_type == "Concat":
        return ConcatFusion(
            dim1=args.encoder_dim,
            dim2=agg_out_dim,
            fusion_dim=args.fusion_dim,
        )
    return AttentionFusion(
        dim1=args.encoder_dim,
        dim2=agg_out_dim,
        fusion_dim=args.fusion_dim,
    )


def build_dataloader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def select_monitor_value(metric_name, metric_dict):
    return metric_dict[metric_name]


def save_run_config(args):
    serializable = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    with open(args.config_path, "w", encoding="utf-8") as fout:
        json.dump(serializable, fout, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    save_run_config(args)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = HatefulMemeDataset(
        args.train_json,
        args.image_dir,
        base_model=args.base_model,
        processor_name=args.encoder_model_name,
    )
    val_dataset = HatefulMemeDataset(
        args.val_json,
        args.image_dir,
        base_model=args.base_model,
        processor_name=args.encoder_model_name,
    )
    test_dataset = HatefulMemeDataset(
        args.test_json,
        args.image_dir,
        base_model=args.base_model,
        processor_name=args.encoder_model_name,
    )

    train_dataloader = build_dataloader(
        train_dataset, args.batch_size, True, args.num_workers
    )
    val_dataloader = build_dataloader(
        val_dataset, args.batch_size, False, args.num_workers
    )
    test_dataloader = build_dataloader(
        test_dataset, args.batch_size, False, args.num_workers
    )

    aggregation_module = build_aggregation_module(args)
    fusion_module = build_fusion_module(args, aggregation_module.out_dim)

    model = HatefulMemeClassifierIB(
        aggregation_module=aggregation_module,
        fusion_module=fusion_module,
        base_model=args.base_model,
        encoder_model_name=args.encoder_model_name,
        ib_beta=args.ib_beta,
        ib_dim=args.ib_dim,
        use_ib=args.use_ib,
    ).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(len(train_dataloader), 1) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_steps,
    )

    best_metric = float("-inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, train_acc, avg_ce, avg_ib = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            feature_path=args.feature_path_train,
            neighbor_path=args.neighbor_path_train,
            log_every_batches=args.log_every,
        )
        val_f1, val_auc, val_acc = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            feature_path=args.feature_path_val,
            neighbor_path=args.neighbor_path_val,
        )
        metric_dict = {"f1": val_f1, "auc": val_auc, "acc": val_acc}
        current_metric = select_monitor_value(args.monitor_metric, metric_dict)

        print(
            "Epoch {} Summary: Train Loss: {:.4f}, Train Acc: {:.4f} | "
            "Val F1: {:.4f}, Val AUC: {:.4f}, Val Acc: {:.4f}".format(
                epoch + 1,
                train_loss,
                train_acc,
                val_f1,
                val_auc,
                val_acc,
            )
        )
        with open(args.log_file, "a", encoding="utf-8") as log:
            log.write(
                "Epoch {}: TrainLoss={:.6f}, CE={:.6f}, IB={:.6f}, TrainAcc={:.4f}, "
                "ValF1={:.4f}, ValAUC={:.4f}, ValAcc={:.4f}, Monitor({})={:.4f}\n".format(
                    epoch + 1,
                    train_loss,
                    avg_ce,
                    avg_ib,
                    train_acc,
                    val_f1,
                    val_auc,
                    val_acc,
                    args.monitor_metric,
                    current_metric,
                )
            )

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            torch.save(model.state_dict(), args.best_model_path)
            print(
                "New best validation {}: {:.4f}. Saved to {}".format(
                    args.monitor_metric,
                    best_metric,
                    args.best_model_path,
                )
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    "Early stopping after {} epochs without improvement.".format(
                        patience_counter
                    )
                )
                break

    print("\n--- Final Testing ---")
    model.load_state_dict(torch.load(args.best_model_path, map_location=device))
    test_f1, test_auc, test_acc = evaluate(
        model=model,
        dataloader=test_dataloader,
        device=device,
        feature_path=args.feature_path_test,
        neighbor_path=args.neighbor_path_test,
    )
    print(f"Test Results: F1={test_f1:.4f}, AUC={test_auc:.4f}, Acc={test_acc:.4f}")
    with open(args.log_file, "a", encoding="utf-8") as log:
        log.write(
            "\nTest Results: F1={:.4f}, AUC={:.4f}, Acc={:.4f}\n".format(
                test_f1, test_auc, test_acc
            )
        )
    print(f"Training finished. Logs saved to {args.log_file}")


if __name__ == "__main__":
    main()