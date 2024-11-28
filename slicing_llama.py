import subprocess
import os

from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if not os.path.exists("tmp"):
    os.makedirs("tmp")

model_without_layers = deepcopy(model)
del model_without_layers.model.layers
del model_without_layers.model.norm
del model_without_layers.lm_head
model_without_layers.save_pretrained("tmp/tmp_embed")
tokenizer.save_pretrained("tmp/tmp_embed")
del model_without_layers

model_without_embedding = deepcopy(model)
del model_without_embedding.model.embed_tokens
del model_without_embedding.lm_head
model_without_embedding.save_pretrained("tmp/tmp_layers")
tokenizer.save_pretrained("tmp/tmp_layers")
del model_without_embedding

model_without_head = deepcopy(model)
del model_without_head.model
model_without_head.save_pretrained("tmp/tmp_head")
tokenizer.save_pretrained("tmp/tmp_head")
del model_without_head

hf2gguf = "llama.cpp-b2144/convert-hf-to-gguf.py"

subprocess.run(["python", hf2gguf, "tmp/tmp_embed/", "--outfile", "tmp/tmp_embed/llama_embed", "--outtype", "f16"])
subprocess.run(["python", hf2gguf, "tmp/tmp_layers/", "--outfile", "tmp/tmp_layers/llama_layers", "--outtype", "f16"])
subprocess.run(["python", hf2gguf, "tmp/tmp_head/", "--outfile", "tmp/tmp_head/llama_head", "--outtype", "f16"])