import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

def filter_empty_strings(example):
    return example["text"].strip() != ""

# 모델과 토크나이저 불러오기
model_name_gpt2 = "openai-community/gpt2"
model_name_gpt2_xl = "openai-community/gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name_gpt2)
model_gpt2 = AutoModelForCausalLM.from_pretrained(model_name_gpt2)
model_gpt2_xl = AutoModelForCausalLM.from_pretrained(model_name_gpt2_xl)

# 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 불러오기
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
filtered_dataset = dataset.filter(filter_empty_strings)
dataloader = DataLoader(filtered_dataset, batch_size=1, shuffle=True)

# 학습 설정
optimizer = AdamW(model_gpt2.parameters(), lr=5e-5)

def kl_divergence_loss(logits1, logits2):
    p = torch.nn.functional.log_softmax(logits1, dim=-1)
    q = torch.nn.functional.softmax(logits2, dim=-1)
    return torch.nn.functional.kl_div(p, q, reduction='batchmean')

model_gpt2.train().cuda()
model_gpt2_xl.eval().cuda()

# 학습 루프
for epoch in range(10):  # 에포크 수는 필요에 따라 조정 가능
    with tqdm(dataloader) as pbar:
        for batch in pbar:
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
            with torch.no_grad():
                outputs_gpt2_xl = model_gpt2_xl(inputs)
            outputs_gpt2 = model_gpt2(inputs)

            loss = kl_divergence_loss(outputs_gpt2.logits, outputs_gpt2_xl.logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            pbar.set_description(str(loss.item()))

    print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")

print("Training completed.")

tokenizer.save_pretrained("./tmp")
model_gpt2.cpu().save_pretrained("./tmp")