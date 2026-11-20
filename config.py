import torch

NUM_CLASSES = 54

device = "mps" if torch.backends.mps.is_available() else "cpu"
