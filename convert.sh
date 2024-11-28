#!/bin/bash

model_path=$1
quantization_type=$2

python /home/smp9898/llama.cpp/llama.cpp/convert_hf_to_gguf.py ${model_path} --outtype ${quantization_type}