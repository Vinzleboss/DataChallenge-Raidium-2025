from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset
import pandas as pd
from scipy import stats

# Section Traitement des données

def load_dataset(dataset_dir):
    dataset_list = []
    # Note: It's very important to load the images in the correct numerical order!
    for image_file in list(sorted(Path(dataset_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png")))):
        dataset_list.append(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE))
    return np.stack(dataset_list, axis=0)

class CTDataset(Dataset):
    def __init__(self, images, masks, indices, transform = None):
        self.images = images        # (2000, 256, 256) uint8
        self.masks = masks          # (2000, 256, 256) uint8
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]         # indice d'origine, aligné avec L
        x = torch.from_numpy(self.images[i]).float().unsqueeze(0) / 255.0   # (1, 256, 256)
        y = torch.from_numpy(self.masks[i]).long()                          # (256, 256)
        if self.transform is not None :
            imgtv = tv_tensors.Image(x)
            masktv = tv_tensors.Mask(y)
            x,y = self.transform(imgtv, masktv)

            return x, y , i
            
        return x, y, i
    
def plot_slice_seg(slice_image, seg):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(slice_image, cmap="gray")
    axes[1].imshow(slice_image, cmap="gray")
    seg_masked = np.ma.masked_where(seg.reshape((256,256)) == 0, (seg.reshape((256,256))))
    axes[1].imshow(seg_masked, cmap="tab20")
    plt.axis("off")


#On cherche les plages d'intensité pour plusieurs classes en vue de réaliser une nouvelle augmentation des données

df = pd.read_csv(r'/Users/vincenzo/DataChallenge_Raidium/data/y_train.csv')
pixels_par_classe = {c : {} for c in range(1,55)}

for i in range(1, 55):
    cols_valides = df.columns[(df == i).any()]
    for col in cols_valides:
        k = np.flatnonzero(df[col].values == i)        
        pixels_par_classe[i][col] = np.stack([k // 256, k % 256], axis=1)


def grayscale_per_class(pixs_par_classe, classe) :
    pixs_imgs = pixs_par_classe[classe]
    niveau = []
    for img in pixs_imgs :
        pixs = pixs_imgs[img]
        arr = cv2.imread(f'/Users/vincenzo/DataChallenge_Raidium/data/train-images/{img}', cv2.IMREAD_GRAYSCALE)
        n = len(pixs)
        niv = 0
        for j in range(len(pixs)):
            niv += int(arr[pixs[j][0]][pixs[j][1]])
        niveau.append(niv/n)

    return niveau
 
for i in range(1, 55):
    niveau = grayscale_per_class(pixels_par_classe, i)
    percentile = np.percentile(niveau, [5, 95])

    print(f"Percentiles pour la classe {i} : {percentile}")

