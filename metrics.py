import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import math
from config import NUM_CLASSES

#Section métriques d'évaluation et train#

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
