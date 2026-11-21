"""Reference for label-noise-hurts (needs torch and torchvision: run it on Barkla, in the experiment image).

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


def accuracy(x, y, xt, yt, noise, seed):
    torch.manual_seed(seed)
    y = y.clone()
    if noise:
        g = torch.Generator().manual_seed(seed)
        flip = torch.rand(len(y), generator=g) < noise
        y[flip] = (y[flip] + torch.randint(1, 10, (int(flip.sum()),), generator=g)) % 10  # always a different class
    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        for idx in torch.randperm(len(x), generator=torch.Generator().manual_seed(seed)).split(128):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()
    with torch.no_grad():
        return (model(xt).argmax(1) == yt).float().mean().item()


if __name__ == "__main__":
    x, y = tensors(sys.argv[1], True)
    xt, yt = tensors(sys.argv[1], False)
    x, y = x[:10_000], y[:10_000]
    clean = [accuracy(x, y, xt, yt, 0.0, s) for s in range(6)]
    noisy = [accuracy(x, y, xt, yt, 0.4, s) for s in range(6)]
    print(f"clean {statistics.mean(clean):.4f}, noisy_40 {statistics.mean(noisy):.4f}")
    lower = all(n < c for c, n in zip(clean, noisy, strict=True))
    print("expectation (noisy lower on every seed):", "holds" if lower else "CHECK")
