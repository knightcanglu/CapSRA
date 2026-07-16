import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, ViltModel, ViltProcessor


BACKBONE_REGISTRY = {
    "clip": {
        "model_name": "openai/clip-vit-base-patch16",
        "feature_dim": 512,
        "processor_cls": CLIPProcessor,
        "model_cls": CLIPModel,
    },
    "vilt": {
        "model_name": "dandelin/vilt-b32-finetuned-vqa",
        "feature_dim": 768,
        "processor_cls": ViltProcessor,
        "model_cls": ViltModel,
    },
}


def get_backbone_spec(base_model):
    key = base_model.lower()
    if key not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported base_model '{base_model}'. Supported: {sorted(BACKBONE_REGISTRY)}"
        )
    return BACKBONE_REGISTRY[key]


def get_default_model_name(base_model):
    return get_backbone_spec(base_model)["model_name"]


def get_feature_dim(base_model):
    return get_backbone_spec(base_model)["feature_dim"]


def build_processor(base_model, model_name=None):
    spec = get_backbone_spec(base_model)
    processor_cls = spec["processor_cls"]
    return processor_cls.from_pretrained(model_name or spec["model_name"])


class FrozenMultimodalBackbone(nn.Module):
    def __init__(self, base_model="clip", model_name=None):
        super().__init__()
        self.base_model = base_model.lower()
        spec = get_backbone_spec(self.base_model)
        self.model_name = model_name or spec["model_name"]
        self.output_dim = spec["feature_dim"]
        self.model = spec["model_cls"].from_pretrained(self.model_name)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward_features(self, inputs, device):
        if self.base_model == "clip":
            outputs = self.model(
                pixel_values=inputs["pixel_values"].to(device),
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
            return (outputs.image_embeds + outputs.text_embeds) / 2

        if self.base_model == "vilt":
            model_inputs = {
                "input_ids": inputs["input_ids"].to(device),
                "attention_mask": inputs["attention_mask"].to(device),
                "pixel_values": inputs["pixel_values"].to(device),
            }
            if "pixel_mask" in inputs:
                model_inputs["pixel_mask"] = inputs["pixel_mask"].to(device)
            if "token_type_ids" in inputs:
                model_inputs["token_type_ids"] = inputs["token_type_ids"].to(device)
            outputs = self.model(**model_inputs)
            return outputs.pooler_output

        raise ValueError(f"Unsupported backbone during forward: {self.base_model}")