"""Reference for softmax-mnist-accuracy (needs torch and torchvision: run it on Barkla, in the experiment image).

python reference.py <folder with torchvision's MNIST>        # a few minutes on a CPU
"""

import statistics
import sys

import torch
from torch import nn
from torchvision import datasets


def tensors(root, train):
    data = datasets.MNIST(root, train=train, download=False)
    return data.data.float().div(255).flatten(1), data.targets.clone()


def accuracy(x, y, xt, yt, seed):
    torch.manual_seed(seed)
    model = nn.Linear(784, 10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        for idx in torch.randperm(len(x), generator=torch.Generator().manual_seed(seed)).split(128):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()
    with torch.no_grad():
        return (model(xt).argmax(1) == yt).float().mean().item()


if __name__ == "__main__":
    x, y = tensors(sys.argv[1], True)
    xt, yt = tensors(sys.argv[1], False)
    runs = [accuracy(x, y, xt, yt, s) for s in range(6)]
    mean = statistics.mean(runs)
    print(f"softmax {mean:.4f} over 6 seeds (sd {statistics.stdev(runs):.4f})")
    print("expectation (0.90 <= mean <= 0.94):", "holds" if 0.9 <= mean <= 0.94 else "FAILS")
