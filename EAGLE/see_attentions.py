import torch
import matplotlib.pyplot as plt
import os

layer_num = 32
head_num = 32

ssm = torch.load("EAGLE/attentions/tensors/ssm/0.pt")

for head in range(head_num):
    plt.imshow(torch.pow(ssm[0, head], 1/3), cmap="Blues")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"EAGLE/attentions/graphs/ssm/{head}.png")
    plt.close()

for layer in range(layer_num):
    if not os.path.exists(f"EAGLE/attentions/graphs/ltm/{layer}"):
        os.makedirs(f"EAGLE/attentions/graphs/ltm/{layer}")
    
    ltm = torch.load(f"EAGLE/attentions/tensors/ltm/{layer}.pt")

    for head in range(head_num):
        plt.imshow(torch.pow(ltm[0, head], 1/3), cmap="Blues")
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"EAGLE/attentions/graphs/ltm/{layer}/{head}.png")
        plt.close()