import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

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

    if opt.MODEL != "pbm":
        raise ValueError("This patch bundle only exposes the CapSRA-enabled Pro-Cap pbm path.")

    from dataset import Multimodal_Data
    import pbm_gat as pbm

    train_set = Multimodal_Data(opt, opt.DATASET, "train")
    test_set = Multimodal_Data(opt, opt.DATASET, "test")

    max_length = opt.LENGTH + opt.CAP_LENGTH
    if opt.ASK_CAP != "":
        max_length += opt.CAP_LENGTH * len(opt.ASK_CAP.split(","))
    if opt.NUM_MEME_CAP > 0:
        max_length += opt.CAP_LENGTH * opt.NUM_MEME_CAP
    if opt.USE_DEMO:
        max_length *= (opt.NUM_SAMPLE * opt.NUM_LABELS + 1)

    label_words = [opt.POS_WORD, opt.NEG_WORD]
    model = getattr(pbm, "build_baseline")(
        label_words,
        max_length,
        model_name="roberta-large",
        neighbor_path=None,
        feature_path=None,
        alpha=opt.ALPHA,
        device=torch.device(f"cuda:{opt.CUDA_DEVICE}"),
    )
    model.set_split_data(opt.train_neighbor_path, opt.train_feature_path)

    train_loader = DataLoader(train_set, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=opt.BATCH_SIZE, shuffle=False, num_workers=2)
    train_for_epoch(opt, model, train_loader, test_loader)