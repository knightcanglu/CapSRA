import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utils.utils import set_seed
from data_handler.dataset import HatefulMemeDataset, collate_fn
from models.layers import GATLayer, TransformerConvLayer, WeightedAverageLayer, GatedFusion, ConcatFusion, AttentionFusion
from models.main_model import HatefulMemeClassifierIB
from train_eval import train_one_epoch, evaluate

def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Modular Hateful Meme Classifier Training (CapSRA)")

    # --- Core Paths ---
    parser.add_argument('--data_dir', type=str, default='MOMENTA-main/HarMeme_V1/Annotations/Harm-C', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory for saving results')
    parser.add_argument('--feature_path_train', type=str, default="harmc_features_momenta/train_fused_feats.pt")
    parser.add_argument('--feature_path_dev', type=str, default="harmc_features_momenta/test_fused_feats.pt")
    parser.add_argument('--neighbor_path_train', type=str, default="top_10_neighbors_covid19_qwen_train.json")
    parser.add_argument('--neighbor_path_dev', type=str, default="top_10_neighbors_covid19_qwen_test.json")

    # --- Model Architecture ---
    # Updated choices to reflect the new layer name
    parser.add_argument('--aggregation_type', type=str, default='GAT', choices=['GAT', 'TransformerConv', 'WeightedAverage'], help='Type of neighbor aggregation module')
    parser.add_argument('--fusion_type', type=str, default='Gated', choices=['Gated', 'Concat', 'Attention'], help='Type of feature fusion module')
    parser.add_argument('--clip_model_name', type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument('--agg_out_dim', type=int, default=512, help='Output dimension for the aggregation module')
    parser.add_argument('--fusion_dim', type=int, default=512, help='Output dimension for the fusion module')

    # --- Training Hyperparameters ---
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--num_warmup_steps', type=int, default=200)
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--log_every', type=int, default=50, help='Log every N batches')

    # --- Module-Specific Parameters ---
    # GAT specific
    parser.add_argument('--gat_hidden_dim', type=int, default=256)
    parser.add_argument('--gat_heads', type=int, default=4)
    # TransformerConv specific
    parser.add_argument('--transformer_heads', type=int, default=4)
    # IB specific
    parser.add_argument('--use_ib', action='store_true', default=True, help='Enable Information Bottleneck')
    parser.add_argument('--no_ib', action='store_false', dest='use_ib', help='Disable Information Bottleneck')
    parser.add_argument('--ib_beta', type=float, default=0.0001)
    parser.add_argument('--ib_dim', type=int, default=256)

    args = parser.parse_args()
    
    # Build full paths from the base directory
    args.train_json = os.path.join(args.data_dir, 'train_fix.jsonl')
    args.val_json = os.path.join(args.data_dir, 'val_fix.jsonl')
    args.test_json = os.path.join(args.data_dir, 'test_fix.jsonl')
    args.image_dir = os.path.join(args.data_dir, 'img')
    args.log_file = os.path.join(args.output_dir, 'training_log.txt')
    args.best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    
    return args

def main():
    """Main function to assemble and run the training pipeline."""
    args = parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_dataset = HatefulMemeDataset(args.train_json, args.image_dir)
    val_dataset = HatefulMemeDataset(args.val_json, args.image_dir)
    test_dataset = HatefulMemeDataset(args.test_json, args.image_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.aggregation_type == 'GAT':
        agg_module = GATLayer(in_dim=512, hidden_dim=args.gat_hidden_dim, out_dim=args.agg_out_dim, heads=args.gat_heads)
    elif args.aggregation_type == 'TransformerConv':
        agg_module = TransformerConvLayer(in_dim=512, out_dim=args.agg_out_dim, n_heads=args.transformer_heads)
    else:
        agg_module = WeightedAverageLayer(in_dim=512, out_dim=args.agg_out_dim)

    if args.fusion_type == 'Gated':
        fusion_module = GatedFusion(dim1=512, dim2=agg_module.out_dim, fusion_dim=args.fusion_dim)
    elif args.fusion_type == 'Concat':
        fusion_module = ConcatFusion(dim1=512, dim2=agg_module.out_dim, fusion_dim=args.fusion_dim)
    else:
        fusion_module = AttentionFusion(dim1=512, dim2=agg_module.out_dim, fusion_dim=args.fusion_dim)

    model = HatefulMemeClassifierIB(
        aggregation_module=agg_module,
        fusion_module=fusion_module,
        clip_model_name=args.clip_model_name,
        ib_beta=args.ib_beta,
        ib_dim=args.ib_dim,
        use_ib=args.use_ib
    ).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_steps)

    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_loss, train_acc, avg_ce, avg_ib = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, device,
            args.feature_path_train, args.neighbor_path_train, args.log_every
        )
        val_f1, val_auc, val_acc = evaluate(
            model, val_dataloader, device, args.feature_path_dev, args.neighbor_path_dev
        )
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
        with open(args.log_file, "a", encoding="utf-8") as log:
            log.write(f"Epoch {epoch+1}: TrainLoss={train_loss:.6f}, CE={avg_ce:.6f}, IB={avg_ib:.6f}, TrainAcc={train_acc:.4f}, ValF1={val_f1:.4f}, ValAUC={val_auc:.4f}, ValAcc={val_acc:.4f}\n")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), args.best_model_path)
            print(f"New best validation accuracy: {best_val_acc:.4f}. Model saved to {args.best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {patience_counter} epochs without improvement.")
                break
    
    print("\n--- Final Testing ---")
    model.load_state_dict(torch.load(args.best_model_path))
    test_f1, test_auc, test_acc = evaluate(
        model, test_dataloader, device, args.feature_path_dev, args.neighbor_path_dev
    )
    print(f"Test Results: F1={test_f1:.4f}, AUC={test_auc:.4f}, Acc={test_acc:.4f}")
    with open(args.log_file, "a", encoding="utf-8") as log:
        log.write(f"\nTest Results: F1={test_f1:.4f}, AUC={test_auc:.4f}, Acc={test_acc:.4f}\n")
    print(f"Training finished. Logs saved to {args.log_file}")

if __name__ == "__main__":
    main()
