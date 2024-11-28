import torch
from eagle.model.cnets import Model
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

ea_model_path = "yuhuili/EAGLE-llama2-chat-7B"
config = AutoConfig.from_pretrained(ea_model_path)
weight = torch.load("../models/llama-2-7b-eagle/pytorch_model.bin")
model = Model(config=config)
model.load_state_dict(weight)
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

print("Draft")

embed_tokens_params = sum(p.numel() * torch.finfo(p.dtype).bits / 8 / 1024 / 1024 for p in model.embed_tokens.parameters())
print(f"embed_tokens 파라미터 사이즈: {embed_tokens_params}")

fc_params = sum(p.numel() * torch.finfo(p.dtype).bits / 8 / 1024 / 1024 for p in model.fc.parameters())
print(f"fc 파라미터 사이즈: {fc_params}")

layers_params = sum(p.numel() * torch.finfo(p.dtype).bits / 8 / 1024 / 1024 for layer in model.layers for p in layer.parameters())
print(f"layers 파라미터 사이즈: {layers_params}")

model_params = sum(p.numel() * torch.finfo(p.dtype).bits / 8 / 1024 / 1024 for p in model.parameters())
print(f"models 파라미터 사이즈: {model_params}")

print("Target")

embed_tokens_params = sum(p.numel() * torch.finfo(p.dtype).bits / 8 / 1024 / 1024 for p in target.model.embed_tokens.parameters())
print(f"embed_tokens 파라미터 사이즈: {embed_tokens_params}")

layers_params = sum(p.numel() * torch.finfo(p.dtype).bits / 8 / 1024 / 1024 for layer in target.model.layers for p in layer.parameters())
print(f"layers 파라미터 사이즈: {layers_params}")

model_params = sum(p.numel() * torch.finfo(p.dtype).bits / 8 / 1024 / 1024 for p in target.model.parameters())
print(f"models 파라미터 사이즈: {model_params}")

import pdb; pdb.set_trace()