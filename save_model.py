import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 로드
model_path = "yuhuili/EAGLE-llama2-chat-7B"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 모델 저장
save_dir = "/home/smp9898/llama.cpp/models/llama-2-7b-eagle"
# tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)