"""Small CNN on digits. PLANTED P6: calls .cuda() with no CPU fallback, so it cannot
run in a CPU-only sandbox (expected verdict NOT_RUN)."""
import torch
from torch import nn
from sklearn.datasets import load_digits


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 6 * 6, 10))

    def forward(self, x):
        return self.body(x)


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    model = Net().cuda()
    x = torch.tensor(X, dtype=torch.float32).view(-1, 1, 8, 8).cuda()
    print("logits", model(x).shape)
