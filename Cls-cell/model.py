import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x, x
