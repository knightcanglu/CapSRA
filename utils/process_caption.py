import argparse
import json
import os

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_PROMPT = (
    "You are a meme analyst. Analyze the following meme and output exactly three lines:\n"
    "Meaning: <Summarize the meme's core intention in one clear sentence>\n"
    "Emotion: <Describe the primary emotion in a few words or a short phrase>\n"
    "Visual: <Briefly depict the scene and any overlaid text>"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LMM-guided captions for CapSRA preprocessing."
    )
    parser.add_argument("--jsonl_file", type=str, required=True, help="Input meme jsonl.")
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file containing caption results.",
    )
    parser.add_argument(
        "--img_folder",
        type=str,
        required=True,
        help="Directory containing meme images.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Vision-language model used for caption generation.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device such as cuda:0 or cpu.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt template used to generate meme-aware captions.",
    )
    return parser.parse_args()


def resolve_image_name(item):
    return item.get("img") or item.get("image") or ""


def load_jsonl_rows(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_model_and_processor(args):
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map={"": args.device},
        cache_dir=args.cache_dir,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )
    return model, processor


def generate_caption(model, processor, image, prompt_text, device, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    chat_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    use_amp = str(device).startswith("cuda")
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    generated_ids = [o[len(i) :] for i, o in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    model, processor = build_model_and_processor(args)

    rows = list(load_jsonl_rows(args.jsonl_file))
    results = []

    for meme in tqdm(rows, desc="Generating captions"):
        image_name = resolve_image_name(meme)
        if not image_name:
            continue
        image_path = os.path.join(args.img_folder, image_name)
        if not os.path.exists(image_path):
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        output_text = generate_caption(
            model=model,
            processor=processor,
            image=image,
            prompt_text=args.prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(
            {
                "meme_id": meme.get("id"),
                "image": image_name,
                "description": output_text,
            }
        )

    with open(args.output_file, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(
        "Done: generated captions for {} samples. Saved to {}".format(
            len(results),
            args.output_file,
        )
    )


if __name__ == "__main__":
    main()