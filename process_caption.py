import os
import json
import argparse
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Batch process memes to generate captions using Qwen2.5-VL-7B.")
parser.add_argument("--jsonl_file", type=str, 
                    help="Input JSONL file containing memes (id and image fields).",
                    default="data/dev.jsonl")
parser.add_argument("--output_file", type=str, 
                    help="Output JSON file to write meme results with captions.",
                    default="processed_memes_captions_only_dev_714.json")
parser.add_argument("--img_folder", type=str, 
                    help="Folder path where meme images are stored.",
                    default="data/")
parser.add_argument("--cache_dir", type=str, default="E:/huggingface_models",
                    help="Cache directory for HuggingFace models.")
parser.add_argument("--device", type=str, default="cuda:1",
                    help="Torch device to run model on, e.g., 'cuda:0' or 'cpu'.")
args = parser.parse_args()

device = args.device
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map={"": device},
    cache_dir=args.cache_dir
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir=args.cache_dir
)

with open(args.jsonl_file, 'r', encoding='utf-8') as f:
    total_samples = sum(1 for _ in f)
print(f"total_samples: {total_samples}")

results = []
valid_samples = 0

with open(args.jsonl_file, 'r', encoding='utf-8') as fin, \
     open(args.output_file, 'w', encoding='utf-8') as fout:

    for line in tqdm(fin, total=total_samples, desc="Generating Captions"):
        try:
            meme = json.loads(line)
        except json.JSONDecodeError:
            continue

        img_filename = meme.get("img")
        if not img_filename:
            continue

        img_path = os.path.join(args.img_folder, img_filename)
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        valid_samples += 1

        prompt_text = (
            "You are a meme analyst. Analyze the following meme and output exactly three lines:\n"
            "Meaning: <Summarize the meme’s core intention in one clear sentence>"
            "Emotion: <Describe the primary emotion in a few words or a short phrase>"
            "Visual: <Briefly depict the scene and any overlaid text>"
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]}
        ]

        chat_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[chat_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )

        gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        output_text = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        print(f"Generated Caption for {img_filename}:\n{output_text}\n")

        meme_result = {
            "meme_id": meme.get("id"),
            "image": img_filename,
            "description": output_text
        }
        results.append(meme_result)

    json.dump(results, fout, ensure_ascii=False, indent=2)

print(f"Done: samples {valid_samples}, output {len(results)} . File: {args.output_file}")
