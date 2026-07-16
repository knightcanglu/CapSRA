import json
import os
import pickle as pkl
import random

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
    def __init__(self, opt, tokenizer, dataset, mode="train", few_shot_index=0):
        super().__init__()
        self.opt = opt
        self.tokenizer = tokenizer
        self.mode = mode
        if self.opt.FEW_SHOT:
            self.few_shot_index = str(few_shot_index)
            self.num_shots = self.opt.NUM_SHOTS

        self.num_ans = self.opt.NUM_LABELS
        self.length = self.opt.LENGTH
        self.total_length = self.opt.TOTAL_LENGTH
        self.num_sample = self.opt.NUM_SAMPLE
        self.add_ent = self.opt.ADD_ENT
        self.add_dem = self.opt.ADD_DEM
        self.fine_grind = self.opt.FINE_GRIND

        if opt.FINE_GRIND:
            if self.opt.DATASET == "mem":
                self.label_mapping_word = {
                    0: "nobody",
                    1: "race",
                    2: "disability",
                    3: "nationality",
                    4: "sex",
                    5: "religion",
                }
            elif self.opt.DATASET == "harm":
                self.label_mapping_word = {
                    0: "nobody",
                    1: "society",
                    2: "individual",
                    3: "community",
                    4: "organization",
                }
                self.attack_list = {"society": 0, "individual": 1, "community": 2, "organization": 3}
                self.attack_file = load_pkl(os.path.join(self.opt.DATA, "domain_splits", "harm_trgt.pkl"))
            self.template = "*<s>**sent_0*.*_It_was_targeting*label_**</s>*"
        else:
            self.label_mapping_word = {0: self.opt.POS_WORD, 1: self.opt.NEG_WORD}
            self.template = "*<s>**sent_0*.*_It_was*label_**</s>*"

        self.label_mapping_id = {}
        for label in self.label_mapping_word.keys():
            assert len(tokenizer.tokenize(" " + self.label_mapping_word[label])) == 1
            self.label_mapping_id[label] = tokenizer._convert_token_to_id(
                tokenizer.tokenize(" " + self.label_mapping_word[label])[0]
            )

        self.template_list = self.template.split("*")
        self.special_token_mapping = {
            "<s>": tokenizer.convert_tokens_to_ids("<s>"),
            "<mask>": tokenizer.mask_token_id,
            "<pad>": tokenizer.pad_token_id,
            "</s>": tokenizer.convert_tokens_to_ids("<\\s>"),
        }

        if self.opt.DEM_SAMP:
            self.img_rate = self.opt.IMG_RATE
            self.text_rate = self.opt.TEXT_RATE
            self.samp_rate = self.opt.SIM_RATE
            self.clip_clean = self.opt.CLIP_CLEAN
            clip_path = os.path.join(self.opt.CAPTION_PATH, dataset, dataset + "_sim_scores.pkl")
            self.clip_feature = load_pkl(clip_path)

        self.support_examples = self.load_entries("train")
        self.entries = self.load_entries(mode)
        if self.opt.DEBUG:
            self.entries = self.entries[:128]
        self.prepare_exp()

    def load_entries(self, mode):
        if self.opt.FEW_SHOT and mode == "train":
            path = os.path.join(
                self.opt.DATA,
                "domain_splits",
                self.opt.DATASET + "_" + str(self.num_shots) + "_" + self.few_shot_index + ".json",
            )
        else:
            path = os.path.join(self.opt.DATA, "domain_splits", self.opt.DATASET + "_" + mode + ".json")
        data = read_json(path)
        cap_path = os.path.join(
            self.opt.CAPTION_PATH,
            self.opt.DATASET + "_" + self.opt.PRETRAIN_DATA,
            self.opt.IMG_VERSION + "_captions.pkl",
        )
        captions = load_pkl(cap_path)
        entries = []
        for row in data:
            label = row["label"]
            img = row["img"]
            cap = captions[img.split(".")[0]][:-1]
            sent = row["clean_sent"]
            cap = cap + " . " + sent + " . "
            if self.add_ent:
                cap = cap + " . " + row["entity"] + " . "
            if self.add_dem:
                cap = cap + " . " + row["race"] + " . "
            entry = {"cap": cap.strip(), "label": label, "img": img}
            if self.fine_grind:
                if self.opt.DATASET == "mem":
                    entry["attack"] = [1] + row["attack"] if label == 0 else [0] + row["attack"]
                elif self.opt.DATASET == "harm":
                    if label == 0:
                        entry["attack"] = [1, 0, 0, 0, 0]
                    else:
                        attack = [0, 0, 0, 0, 0]
                        attack_idx = self.attack_list[self.attack_file[img]] + 1
                        attack[attack_idx] = 1
                        entry["attack"] = attack
            entries.append(entry)
        return entries

    def enc(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prepare_exp(self):
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in tqdm(range(self.num_sample)):
            for query_idx in range(len(self.entries)):
                if self.opt.DEM_SAMP:
                    candidates = [
                        support_idx
                        for support_idx in support_indices
                        if support_idx != query_idx or self.mode != "train"
                    ]
                    sim_score = []
                    count_each_label = {label: 0 for label in range(self.opt.NUM_LABELS)}
                    context_indices = []
                    clip_info_que = self.clip_feature[self.entries[query_idx]["img"]]
                    for support_idx in candidates:
                        img = self.support_examples[support_idx]["img"]
                        img_sim = clip_info_que["clean_img"][img] if self.clip_clean else clip_info_que["img"][img]
                        text_sim = clip_info_que["text"][img]
                        total_sim = self.img_rate * img_sim + self.text_rate * text_sim
                        sim_score.append((support_idx, total_sim))
                    sim_score.sort(key=lambda x: x[1], reverse=True)
                    num_valid = int(len(sim_score) // self.opt.NUM_LABELS * self.samp_rate)
                    for support_idx, _ in sim_score:
                        cur_label = self.support_examples[support_idx]["label"]
                        if count_each_label[cur_label] < num_valid:
                            count_each_label[cur_label] += 1
                            context_indices.append(support_idx)
                else:
                    context_indices = [
                        support_idx
                        for support_idx in support_indices
                        if support_idx != query_idx or self.mode != "train"
                    ]
                self.example_idx.append((query_idx, context_indices, sample_idx))

    def select_context(self, context_examples):
        num_labels = self.opt.NUM_LABELS
        max_demo_per_label = 1
        counts = {k: 0 for k in range(num_labels)}
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

    def process_prompt(self, examples, first_sent_limit, other_sent_limit):
        prompt_arch = " It was targeting " if self.fine_grind else " It was "
        input_ids = []
        attention_mask = []
        concat_sent = ""
        for segment_id, ent in enumerate(examples):
            new_tokens = []
            if segment_id == 0:
                new_tokens.append(self.special_token_mapping["<s>"])
                length = first_sent_limit
                temp = prompt_arch + "<mask>" + " . </s>"
            else:
                length = other_sent_limit
                if self.fine_grind:
                    if ent["label"] == 0:
                        label_word = self.label_mapping_word[0]
                    else:
                        attack_types = [i for i, x in enumerate(ent["attack"]) if x == 1]
                        attack_idx = random.randint(1, 5) if len(attack_types) == 0 else attack_types[np.random.permutation(len(attack_types))[0]]
                        label_word = self.label_mapping_word[attack_idx]
                else:
                    label_word = self.label_mapping_word[ent["label"]]
                temp = prompt_arch + label_word + " . </s>"
            new_tokens += self.enc(" " + ent["cap"])
            new_tokens = new_tokens[:length]
            new_tokens += self.enc(temp)
            concat_sent += " " + ent["cap"] + temp
            input_ids += new_tokens
            attention_mask += [1 for _ in range(len(new_tokens))]

        while len(input_ids) < self.total_length:
            input_ids.append(self.special_token_mapping["<pad>"])
            attention_mask.append(0)
        if len(input_ids) > self.total_length:
            input_ids = input_ids[: self.total_length]
            attention_mask = attention_mask[: self.total_length]
        mask_pos = [input_ids.index(self.special_token_mapping["<mask>"])]
        return {
            "input_ids": input_ids,
            "sent": "<s>" + concat_sent,
            "attention_mask": attention_mask,
            "mask_pos": mask_pos,
        }

    def __getitem__(self, index):
        entry = self.entries[index]
        query_idx, context_indices, _ = self.example_idx[index]
        supports = self.select_context([self.support_examples[i] for i in context_indices])
        exps = [entry]
        exps.extend(supports)
        prompt_features = self.process_prompt(exps, self.length, self.length)
        vid = entry["img"]
        label = torch.tensor(entry["label"])
        target = torch.from_numpy(np.zeros((self.num_ans), dtype=np.float32))
        target[label] = 1.0
        batch = {
            "sent": prompt_features["sent"],
            "mask": torch.Tensor(prompt_features["attention_mask"]),
            "img": vid,
            "target": target,
            "cap_tokens": torch.Tensor(prompt_features["input_ids"]),
            "mask_pos": torch.LongTensor(prompt_features["mask_pos"]),
            "label": label,
        }
        if self.fine_grind:
            batch["attack"] = torch.Tensor(entry["attack"])
        return batch

    def __len__(self):
        return len(self.entries)