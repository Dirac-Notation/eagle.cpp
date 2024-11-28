import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

layer_num = 32
head_num = 32

ssm = torch.load("EAGLE/attentions/tensors/ssm/0.pt").squeeze(0).to(torch.float)
ltm = torch.stack([torch.load(f"EAGLE/attentions/tensors/ltm/{layer}.pt") for layer in range(layer_num)]).squeeze(1).to(torch.float)
import pdb; pdb.set_trace()
if ssm.size(-2) != ltm.size(-2):
    assert "ssm and ltm dimension problem"

mul = torch.arange(ssm.size(-2)) + 1
mul = mul.view(1, -1, 1)

ssm = ssm * mul
ltm = ltm * mul.unsqueeze(0)

mul = mul.view(-1).flip(dims=[0])

ssm_path = os.path.join("EAGLE/attentions/h2o", "ssm")
ltm_path = os.path.join("EAGLE/attentions/h2o", "ltm")

for path in [ssm_path, ltm_path]:
   if not os.path.exists(path):
       os.makedirs(path)

for i in tqdm(range(head_num)):
    plt.plot(torch.sum(ssm[i]/mul, dim=0))
    # plt.plot(torch.sum(ssm[i], dim=0))
    plt.yscale("log", base=10)
    plt.savefig(f"{ssm_path}/{i}.png")
    plt.close()

for i in tqdm(range(layer_num)):
    for j in range(head_num):
        tmp_ltm_path = os.path.join(ltm_path, str(i))
        if not os.path.exists(tmp_ltm_path):
            os.makedirs(tmp_ltm_path)
        plt.plot(torch.sum(ltm[i,j]/mul, dim=0))
        # plt.plot(torch.sum(ltm[i,j], dim=0))
        plt.yscale("log", base=10)
        plt.savefig(f"{tmp_ltm_path}/{j}.png")
        plt.close()