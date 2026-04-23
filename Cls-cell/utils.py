import random

import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def accuracy(probabilities, labels):
    predictions = torch.argmax(probabilities, dim=1)
    return (predictions == labels).float().mean().item()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
