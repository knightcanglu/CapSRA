import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class HatefulMemeDataset(Dataset):
    def __init__(self, data_file, image_dir, processor_name="openai/clip-vit-base-patch16"):
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
        self.processor = CLIPProcessor.from_pretrained(processor_name)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item.get('image', ""))
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: cannot open image {image_path}: {e}. Using a blank image.")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
            
        text = item.get('text', "")
        meme_id = item.get('id', "")
        
        image = image_transform(image)
        text_inputs = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        label = torch.tensor(item.get('labels', 0), dtype=torch.long)
        
        inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": image
        }
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
    labels = torch.tensor(list(labels), dtype=torch.long)
    
    return meme_ids, inputs, labels
