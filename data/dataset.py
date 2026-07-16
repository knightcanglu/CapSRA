import os
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from models.backbones import build_processor, get_default_model_name

class HatefulMemeDataset(Dataset):
    def __init__(
        self,
        data_file,
        image_dir,
        base_model="clip",
        processor_name=None,
    ):
        self.data = []
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {data_file}: {line}")
                        continue
        except Exception as e:
            print(f"Error loading data_file {data_file}: {e}")
            
        self.image_dir = image_dir
        self.base_model = base_model.lower()
        self.processor_name = processor_name or get_default_model_name(self.base_model)
        self.processor = build_processor(self.base_model, self.processor_name)

    @staticmethod
    def _resolve_image_name(item):
        return item.get("image") or item.get("img") or ""

    @staticmethod
    def _resolve_text(item):
        return item.get("text") or item.get("caption") or item.get("description") or ""

    @staticmethod
    def _resolve_label(item):
        if "labels" in item:
            return item["labels"]
        if "label" in item:
            return item["label"]
        return 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_name = self._resolve_image_name(item)
        image_path = os.path.join(self.image_dir, image_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: cannot open image {image_path}: {e}. Using a blank image.")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
            
        text = self._resolve_text(item)
        meme_id = item.get("id", "")

        if self.base_model == "clip":
            processed = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
        elif self.base_model == "vilt":
            processed = self.processor(
                image,
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
        else:
            raise ValueError(f"Unsupported base_model: {self.base_model}")
        processed = {k: v.squeeze(0) for k, v in processed.items()}
        label = torch.tensor(self._resolve_label(item), dtype=torch.long)
        
        inputs = {
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "pixel_values": processed["pixel_values"],
        }
        if "pixel_mask" in processed:
            inputs["pixel_mask"] = processed["pixel_mask"]
        if "token_type_ids" in processed:
            inputs["token_type_ids"] = processed["token_type_ids"]
        return meme_id, inputs, label

def collate_fn(batch):
    meme_ids, inputs_list, labels = zip(*batch)
    meme_ids = list(meme_ids)
    
    input_ids = pad_sequence([x["input_ids"] for x in inputs_list], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([x["attention_mask"] for x in inputs_list], batch_first=True, padding_value=0)
    pixel_values = torch.stack([x["pixel_values"] for x in inputs_list])
    
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values
    }
    if "pixel_mask" in inputs_list[0]:
        inputs["pixel_mask"] = torch.stack([x["pixel_mask"] for x in inputs_list])
    if "token_type_ids" in inputs_list[0]:
        inputs["token_type_ids"] = pad_sequence(
            [x["token_type_ids"] for x in inputs_list],
            batch_first=True,
            padding_value=0,
        )
    labels = torch.tensor(list(labels), dtype=torch.long)
    
    return meme_ids, inputs, labels