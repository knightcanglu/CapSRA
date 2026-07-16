import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

import config
from train import train_for_epoch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    opt = config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    set_seed(opt.SEED)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    if opt.MODEL != "pbm":
        raise ValueError("This patch bundle only exposes the CapSRA-enabled PromptHate pbm path.")

    from dataset import Multimodal_Data
    import baselineGAT

    train_set = Multimodal_Data(opt, tokenizer, opt.DATASET, "train", opt.SEED - 1111)
    test_set = Multimodal_Data(opt, tokenizer, opt.DATASET, "test")

    train_loader = DataLoader(
        train_set,
        batch_size=opt.BATCH_SIZE,
        shuffle=True,
        num_workers=opt.NUM_WORKERS if hasattr(opt, "NUM_WORKERS") else 1,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=opt.BATCH_SIZE,
        shuffle=False,
        num_workers=opt.NUM_WORKERS if hasattr(opt, "NUM_WORKERS") else 1,
    )

    label_list = [train_set.label_mapping_id[i] for i in train_set.label_mapping_word.keys()]
    model = baselineGAT.build_baseline_with_gat(opt, label_list).cuda()
    train_for_epoch(opt, model, train_loader, test_loader)