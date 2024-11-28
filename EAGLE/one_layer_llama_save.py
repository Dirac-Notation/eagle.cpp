import torch
from eagle.model.cnets import Model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

model_name = "meta-llama/Llama-2-7b-hf"

big = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
big.model.layers = big.model.layers[:1]

tokenizer.save_pretrained("../models/llama-2-7b-small")
big.save_pretrained("../models/llama-2-7b-small")