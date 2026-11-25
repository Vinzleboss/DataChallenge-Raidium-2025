import json
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from config import (Y_TRAIN_CSV, TRAIN_IMAGES_DIR, ANNOTATED_LABELS_JSON, MODELS_DIR, FIGURES_DIR, MODEL_PATH,
                    N_ANNOTATED, TRAIN_VAL_SPLIT, AUG_DEGREES, AUG_SCALE, EPOCHS, LR, WEIGHT_DECAY,
                    BATCH_SIZE_TRAIN, BATCH_SIZE_VAL, SPLIT_SEED, TORCH_SEED,CONF, TEACHER_PATH, N_ALL, device)
from model import UnetCNN
from losses import cross_entropy_modified
from data import load_dataset, CTDataset
from train import train
from tqdm import tqdm

## LOADING DONNÉES

##Entrainement sur le dataset restreint de 800 images partiellement anotées 
"""
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
gen = torch.Generator().manual_seed(SPLIT_SEED)
train_setnoT, val_set = random_split(fullNoT, TRAIN_VAL_SPLIT, generator=gen)

train_set = CTDataset(data_train, masks, indices = train_setnoT.indices, transform = transformtot)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_VAL, shuffle=False)

"""
## Entraînement sur la totalité du dataset d'images

#Chargement du modèle professeur (le meilleur que l'on ait. Pour l'instant DICE SCORE : 0.332 d'après évaluation sur la plateforme du challenge)
with open(ANNOTATED_LABELS_JSON) as f :
    L = json.load(f)
L_2ndphase = L
data_train = load_dataset(TRAIN_IMAGES_DIR)
labels_train = pd.read_csv(Y_TRAIN_CSV, index_col=0)
masks_2ndphase = labels_train.values.T.astype(np.uint8).reshape(-1,256,256)
model = UnetCNN().to(device)
model.load_state_dict(torch.load(TEACHER_PATH, map_location=device, weights_only=True))
del labels_train
#Chargement des données que l'on va évaluer
images_non_label = data_train[800: ]

#Copie de la fonction de prediction 

@torch.no_grad()
def predict_pseudo(model, images, device, batch_size=16):
    model.eval()
    pseudo_mask = []
    for start in tqdm(range(0, len(images), batch_size), desc="Prédiction test"):
        x = torch.from_numpy(images[start:start + batch_size]).float().unsqueeze(1) / 255.0
        logits = model(x.to(device))
        proba = F.softmax(logits, dim=1)
        confiance, classe = torch.max(proba, dim=1)
        pseudo_mask_batch = torch.where(confiance >= CONF, classe, torch.zeros_like(classe))
        pseudo_mask.append(pseudo_mask_batch.to(torch.uint8).cpu())
    return torch.cat(pseudo_mask).numpy()
torch.manual_seed(TORCH_SEED)

#Construction des pseudo_masks

pseudo_masks = predict_pseudo(model, images_non_label, device)
print(pseudo_masks.shape)
print((pseudo_masks > 0).mean())
masks_2ndphase[800: ] = pseudo_masks

#Préparation de l'entraînement

transform1 = v2.RandomAffine(degrees = AUG_DEGREES, scale=AUG_SCALE, fill = 0)
transform2 = v2.RandomApply([v2.ColorJitter(brightness=(0.8,1.12), contrast=(0.8,1.15))], p = 0.5)
transformtot = v2.Compose([transform1, 
                           transform2,])

fullNoT = CTDataset(data_train, masks_2ndphase, range(N_ANNOTATED), transform=None)
gen = torch.Generator().manual_seed(SPLIT_SEED)
train_setnoT, val_set = random_split(fullNoT, TRAIN_VAL_SPLIT, generator=gen)

train_indices = list(train_setnoT.indices) + list(range(N_ANNOTATED, N_ALL))
train_set = CTDataset(data_train, masks_2ndphase, indices=train_indices, transform=transformtot)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_VAL, shuffle=False)

if __name__=="__main__" :
    if MODEL_PATH.exists():
        raise FileExistsError(f"{MODEL_PATH} existe déjà : change EXPERIMENT_NAME dans config.py pour ne pas l'écraser.")
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    torch.manual_seed(TORCH_SEED)
    model = UnetCNN()
    train(model, train_loader, val_loader, cross_entropy_modified, L, device=device, epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY)
