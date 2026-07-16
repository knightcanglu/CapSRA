import os
import torch
import torch.nn as nn


class KLDivergence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prob, logits):
        batch_size, _ = prob.shape
        pred_prob = logits
        loss = -prob * torch.log(pred_prob)
        loss = torch.sum(loss, -1)
        return torch.sum(loss) / batch_size


def assert_exits(path):
    assert os.path.exists(path), f"Does not exist : {path}"


def equal_info(a, b):
    assert len(a) == len(b), "File info not equal!"


def same_question(a, b):
    assert a == b, "Not the same question!"


class Logger:
    def __init__(self, output_dir):
        dirname = os.path.dirname(output_dir)
        if dirname and not os.path.exists(dirname):
            os.mkdir(dirname)
        self.log_file = open(output_dir, "w")

    def write(self, msg):
        self.log_file.write(msg + "\n")
        self.log_file.flush()
        print(msg)