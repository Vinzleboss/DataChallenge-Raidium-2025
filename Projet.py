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

NUM_CLASSES = 54



class UnetCNN(nn.Module) :
    def __init__(self, num_classes = 55):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)  
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.down4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels = 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels= 32, kernel_size = 3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        self.up3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.finalconv = nn.Conv2d(in_channels = 16, out_channels=num_classes, kernel_size=1, stride=1)


    def forward(self, x):
        x = self.down1(x)
        x_concatdown1 = x
        x = self.pool(x)

        x = self.down2(x)
        x_concatdown2 = x
        x = self.pool(x)

        x = self.down3(x)
        x_concatdown3 = x
        x = self.pool(x)

        x = self.down4(x)
        x = self.upsample(x)
        x = torch.cat([x,x_concatdown3], dim=1)

        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, x_concatdown2], dim=1)

        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x_concatdown1], dim=1)

        x = self.up3(x)
        x = self.finalconv(x)
        return x
    

def dice_image(prediction, ground_truth) :
    intersection = np.sum(prediction*ground_truth)
    if np.sum(prediction) == 0 and np.sum(ground_truth) ==0:
        return np.nan
    return 2*intersection/(np.sum(prediction) + np.sum(ground_truth))

def dice_multiclass(prediction, ground_truth, A) :
    dices = []
    for i in range(1,NUM_CLASSES+1):
        if i in A : 
            dices.append(dice_image(prediction==i, ground_truth==i))
        else :
            dices.append(np.nan)

    return np.array(dices)

def dice_pandas(y_true_df : pd.DataFrame, y_pred_df : pd.DataFrame, L) -> float :
    y_pred_df = y_pred_df.T
    y_true_df = y_true_df.T

    individual_dice = []
    for row_index in range(y_true_df.values.shape[0]):
        dices = dice_multiclass(y_true_df.values[row_index].ravel(), y_pred_df.values[row_index].ravel(), L[row_index])
        individual_dice.append(dices)

    final = np.stack(individual_dice)
    cls_dices = np.nanmean(final, axis=0)
    return float(np.nanmean(cls_dices))


@torch.no_grad()
def evaluate_dice(model, loader, lossfunc, L, device) :
    model.eval()
    tot_loss, tot_dice, n = 0.0, 0.0, 0
    
    ypred = []
    A_val = []
    y_true = []

    for x, y, idx in loader :
        A_batch = [L[i] for i in idx.tolist()]
        A_val.extend(A_batch)
        x,y = x.to(device), y.to(device)
        b = y.size(0)
        y_true.append(y.to(torch.uint8).to('cpu'))
        logits = model(x)
        ypred.append(logits.argmax(dim=1).to(torch.uint8).to('cpu'))
        tot_loss += lossfunc(logits,y, A_batch).item()*b
        n+=b

            

    ypred = torch.cat(ypred)
    y_true = torch.cat(y_true)

    ypred = ypred.numpy()
    y_true = y_true.numpy()

    ypred_pd = pd.DataFrame(ypred.reshape(len(A_val), -1)).T
    ytrue_pd = pd.DataFrame(y_true.reshape(len(A_val), -1)).T

    return tot_loss/n, dice_pandas(ytrue_pd, ypred_pd, A_val)




def train(model, train_loader, val_loader, lossfunc,L, device='mps', epochs = 5, lr = 1e-3, wd = 1e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = wd)
    best,best_state = -math.inf, None
    val_losses = []
    train_losses = []
    val_dices = []
    epochses = [i for i in range (epochs)]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for x, y, idx in pbar :
            A_batch = [L[i] for i in idx.tolist()]
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = lossfunc(logits, y, A_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item() : .3f}")

        scheduler.step()
        val_loss, val_dice = evaluate_dice(model, val_loader, lossfunc, L, device)
        train_loss, train_dice = evaluate_dice(model, train_loader, lossfunc, L, device)
        print("VALIDATION DICE SCORE : ", val_dice)
        print("VALIDATION LOSS : ", val_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        train_losses.append(train_loss)
    

        if val_dice > best:
            best = val_dice
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save(best_state, "Best_Model.pt")
    
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochses, train_losses, label="Entraînement")
    axes[0].plot(epochses, val_losses, label="Validation")
    axes[0].set_title("Perte")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochses, val_dices, label="Validation")
    axes[1].set_title("DICE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("Coubes_de_convergence.png")
    
    if best_state is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in best_state.items()}
        )



    

def cross_entropy_modified(logits, y, A):
    B = logits.shape[0]
    A_mask = torch.zeros(B,NUM_CLASSES+1, dtype=torch.bool, device = logits.device)
    for i in range(B):
        A_mask[i, A[i]] = True

    A_maskextend = A_mask[:,:,None,None]
    logits_Abarre = logits.masked_fill(A_maskextend,float('-inf'))
    lse_Abarre = torch.logsumexp(logits_Abarre, dim=1)
    lse_all = torch.logsumexp(logits, dim=1)
    neglogSall = lse_all - lse_Abarre

    y0 = y == 0
    neglogS = torch.sum(neglogSall*y0, dim=(1,2))
    weights = torch.sum(y0, dim=(1,2))
    modifiedentropy = 1/weights.clamp(min=1) * neglogS

    ydiff0 = y !=0
    classicentropall = F.cross_entropy(logits,y, reduction="none")

    classicentrop = torch.sum(classicentropall*ydiff0, dim=(1,2))
    weightsent = torch.sum(ydiff0, dim=(1,2))
    entrop = 1/weightsent.clamp(min=1) * classicentrop

    lossall = entrop + modifiedentropy
    loss = torch.sum(lossall) * 1/B

    return loss


## LOADING DONNÉES



labels_train = pd.read_csv("y_train.csv", index_col=0)

def load_dataset(dataset_dir):
    dataset_list = []
    # Note: It's very important to load the images in the correct numerical order!
    for image_file in list(sorted(Path(dataset_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png")))):
        dataset_list.append(cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE))
    return np.stack(dataset_list, axis=0)

data_dir = Path("./")
data_train = load_dataset(data_dir / "train-images")
data_test = load_dataset(data_dir / "test-images")

with open("annotated_labels.json") as f :
    L = json.load(f)

L_1stphase = L[0:800]



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

device = "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(0)

if __name__=="__main__" :
    model = UnetCNN()
    train(model, train_loader, val_loader, cross_entropy_modified, L, device=device, epochs=100)



