from skimage.segmentation import watershed
from skimage.filters import sobel
import pandas as pd
from pathlib import Path
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.filters import rank
from scipy import ndimage as ndi
from skimage.morphology import disk
import sklearn.metrics
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torchvision import tv_tensors
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import math
import json
from torch.utils.data import Dataset
from config import *
from model import *
from metrics import *
from losses import *
from data import *

## LOADING DONNÉES



labels_train = pd.read_csv("y_train.csv", index_col=0)

data_dir = Path("./")
data_train = load_dataset(data_dir / "train-images")
data_test = load_dataset(data_dir / "test-images")

with open("annotated_labels.json") as f :
    L = json.load(f)

L_1stphase = L[0:800]

masks = labels_train.values.T.astype(np.uint8).reshape(-1, 256, 256)
del labels_train
# --- Découpage des 800 images annotées

transform = v2.RandomAffine(degrees = 15, scale=(0.7,1.4), fill = 0)
fullNoT = CTDataset(data_train, masks, range(800), transform=None)
fullT = CTDataset(data_train, masks, range(800), transform=transform)
gen = torch.Generator().manual_seed(42)
train_setnoT, val_set = random_split(fullNoT, [640,160], generator=gen)

train_set = CTDataset(data_train, masks, indices = train_setnoT.indices, transform = transform)







train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

torch.manual_seed(0)

if __name__=="__main__" :
    model = UnetCNN()
    train(model, train_loader, val_loader, cross_entropy_modified, L, device=device, epochs=100)
