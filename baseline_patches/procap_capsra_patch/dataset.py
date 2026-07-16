import json
import os
import pickle as pkl

import numpy as np
import torch
from tqdm import tqdm

import utils


def load_pkl(path):
    return pkl.load(open(path, "rb"))


def read_json(path):
    utils.assert_exits(path)
    return json.load(open(path, "rb"))


class Multimodal_Data:
    def __init__(self, opt, dataset, mode="train"):
        super().__init__()
        self.opt = opt
        self.mode = mode
        self.num_ans = self.opt.NUM_LABELS
        self.num_sample = self.opt.NUM_SAMPLE
        self.ask_cap = len(self.opt.ASK_CAP.split(",")) >= 1
        self.add_ent = self.opt.ADD_ENT
        self.add_dem = self.opt.ADD_DEM
        self.num_meme_cap = self.opt.NUM_MEME_CAP
        self.label_mapping_word = {0: self.opt.POS_WORD, 1: self.opt.NEG_WORD}
        self.template = "*<s>**sent_0*.*_It_was*label_**</s>*"
        self.support_examples = self.load_entries("train")
        self.entries = self.load_entries(mode)
        if self.opt.DEBUG:
            self.entries = self.entries[:128]
        self.prepare_exp()

    def load_entries(self, mode):
        path = os.path.join(self.opt.DATA, self.opt.DATASET + "_" + mode + ".json")
        data = read_json(path)
        if self.opt.CAP_TYPE == "caption":
            cap_path = os.path.join(
                self.opt.CAPTION_PATH,
                self.opt.DATASET + "_" + self.opt.PRETRAIN_DATA,
                self.opt.IMG_VERSION + "_captions.pkl",
            )
        elif self.opt.CAP_TYPE == "vqa":
            cap_path = os.path.join("../../Ask-Captions/Captions", self.opt.DATASET, mode + "_generic.pkl")
            if self.opt.ASK_CAP != "":
                questions = self.opt.ASK_CAP.split(",")
                result_files = {
                    q: load_pkl(os.path.join("../../Ask-Captions/" + self.opt.LONG + "Captions", self.opt.DATASET, mode + "_" + q + ".pkl"))
                    for q in questions
                }
                for v in ["valid_person", "valid_animal"]:
                    result_files[v] = load_pkl(
                        os.path.join("../../Ask-Captions/" + self.opt.LONG + "Captions", self.opt.DATASET, mode + "_" + v + ".pkl")
                    )

        captions = load_pkl(cap_path)
        entries = []
        for row in data:
            label = row["label"]
            img = row["img"]
            if self.opt.CAP_TYPE == "caption":
                cap = captions[img.split(".")[0]][:-1]
            elif self.opt.CAP_TYPE == "vqa" and self.ask_cap:
                cap = captions[img]
                ext = []
                person_flag = not result_files["valid_person"][row["img"]].lower().startswith("no")
                animal_flag = not result_files["valid_animal"][row["img"]].lower().startswith("no")
                for q in questions:
                    if not person_flag and q in ["race", "gender", "country", "valid_disable"]:
                        continue
                    if not animal_flag and q == "animal":
                        continue
                    if q in ["valid_person", "valid_animal"]:
                        continue
                    info = result_files[q][row["img"]]
                    if q == "valid_disable":
                        if info.startswith("no"):
                            continue
                        ext.append("there is a disabled person")
                    else:
                        ext.append(info)
                if self.num_meme_cap > 0:
                    pnp_cap_path = os.path.join("../../Ask-Captions/pnp-captions", self.opt.DATASET, img + ".json")
                    if os.path.exists(pnp_cap_path):
                        caps = read_json(pnp_cap_path)
                        ext.extend(caps[: self.num_meme_cap])
                    else:
                        ext.extend([cap] * self.num_meme_cap)
                cap = cap + " . " + " . ".join(ext)
            else:
                cap = captions[img]
                ext = []
                if self.num_meme_cap > 0:
                    pnp_cap_path = os.path.join("../../Ask-Captions/pnp-captions", self.opt.DATASET, img + ".json")
                    if os.path.exists(pnp_cap_path):
                        caps = read_json(pnp_cap_path)
                        ext.extend(caps[: self.num_meme_cap])
                    else:
                        ext.extend([cap] * self.num_meme_cap)
                cap = cap + " . " + " . ".join(ext)

            sent = row["clean_sent"]
            if self.add_ent:
                cap = cap + " . " + row["entity"] + " . "
            if self.add_dem:
                cap = cap + " . " + row["race"] + " . "
            entries.append(
                {
                    "cap": cap.strip(),
                    "meme_text": sent,
                    "label": label,
                    "img": img,
                }
            )
        return entries

    def prepare_exp(self):
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in tqdm(range(self.num_sample)):
            for query_idx in range(len(self.entries)):
                context_indices = [
                    support_idx for support_idx in support_indices if support_idx != query_idx or self.mode != "train"
                ]
                self.example_idx.append((query_idx, context_indices, sample_idx))

    def select_context(self, context_examples):
        max_demo_per_label = 1
        counts = {k: 0 for k in range(self.opt.NUM_LABELS)}
        selection = []
        order = np.random.permutation(len(context_examples))
        for i in order:
            label = context_examples[i]["label"]
            if counts[label] < max_demo_per_label:
                selection.append(context_examples[i])
                counts[label] += 1
            if sum(counts.values()) == len(counts) * max_demo_per_label:
                break
        assert len(selection) > 0
        return selection

    def process_prompt(self, examples):
        prompt_arch = " It was "
        concat_sent = []
        for segment_id, ent in enumerate(examples):
            temp = prompt_arch + ("<mask> . " if segment_id == 0 else self.label_mapping_word[ent["label"]] + " . ")
            whole_sent = ent["meme_text"] + " . " + temp + ent["cap"]
            concat_sent.append(whole_sent)
            if segment_id == 0:
                test_text = ent["meme_text"] + " . " + ent["cap"]
        return concat_sent, test_text

    def __getitem__(self, index):
        entry = self.entries[index]
        _, context_indices, _ = self.example_idx[index]
        supports = self.select_context([self.support_examples[i] for i in context_indices])
        exps = [entry]
        exps.extend(supports)
        concate_sent, test_text = self.process_prompt(exps)
        prompt_texts = " . </s> ".join(concate_sent)
        label = torch.tensor(entry["label"])
        target = torch.from_numpy(np.zeros((self.num_ans), dtype=np.float32))
        target[label] = 1.0
        return {
            "img": entry["img"],
            "target": target,
            "test_all_text": concate_sent[0] + " . </s> ",
            "test_text": test_text,
            "prompt_all_text": prompt_texts,
            "label": label,
        }

    def __len__(self):
        return len(self.entries)