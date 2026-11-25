import json
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split

from config import (Y_TRAIN_CSV, TRAIN_IMAGES_DIR, ANNOTATED_LABELS_JSON, MODELS_DIR, FIGURES_DIR, MODEL_PATH,
                    N_ANNOTATED, TRAIN_VAL_SPLIT, AUG_DEGREES, AUG_SCALE, EPOCHS, LR, WEIGHT_DECAY,
                    BATCH_SIZE_TRAIN, BATCH_SIZE_VAL, SPLIT_SEED, TORCH_SEED, device)
from model import UnetCNN
from losses import cross_entropy_modified
from data import load_dataset, CTDataset
from train import train

## LOADING DONNÉES



labels_train = pd.read_csv(Y_TRAIN_CSV, index_col=0)

data_train = load_dataset(TRAIN_IMAGES_DIR)

with open(ANNOTATED_LABELS_JSON) as f :
    L = json.load(f)

L_1stphase = L[0:N_ANNOTATED]

masks = labels_train.values.T.astype(np.uint8).reshape(-1, 256, 256)
del labels_train
# --- Découpage des 800 images annotées

transform1 = v2.RandomAffine(degrees = AUG_DEGREES, scale=AUG_SCALE, fill = 0)
transform2 = v2.RandomApply([v2.ColorJitter(brightness=(0.8,1.12), contrast=(0.8,1.15))], p = 0.5)
transformtot = v2.Compose([transform1, 
                           transform2,])
fullNoT = CTDataset(data_train, masks, range(N_ANNOTATED), transform=None)
fullT = CTDataset(data_train, masks, range(N_ANNOTATED), transform=transformtot)
gen = torch.Generator().manual_seed(SPLIT_SEED)
train_setnoT, val_set = random_split(fullNoT, TRAIN_VAL_SPLIT, generator=gen)

train_set = CTDataset(data_train, masks, indices = train_setnoT.indices, transform = transformtot)







train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_VAL, shuffle=False)

torch.manual_seed(TORCH_SEED)

if __name__=="__main__" :
    if MODEL_PATH.exists():
        raise FileExistsError(f"{MODEL_PATH} existe déjà : change EXPERIMENT_NAME dans config.py pour ne pas l'écraser.")
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    model = UnetCNN()
    train(model, train_loader, val_loader, cross_entropy_modified, L, device=device, epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY)
