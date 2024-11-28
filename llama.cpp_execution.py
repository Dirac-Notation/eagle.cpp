from llama_cpp import Llama

model_path = "/home/smp9898/llama.cpp/models/llama-2-7b/Llama-2-7B-hf-F16.gguf"

llm = Llama(model_path=model_path, n_gpu_layers=32)

text = "What is your name?"

answer = llm(text)

print(answer["choices"][0]["text"])