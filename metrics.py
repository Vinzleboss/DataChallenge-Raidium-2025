import pandas as pd
import numpy as np
import torch
from config import NUM_CLASSES

#Section métriques d'évaluation

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
