from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import math
from config import MODEL_PATH, CURVES_PATH
from metrics import evaluate_dice

# Section entraînement

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            torch.save(best_state, MODEL_PATH)
    
    
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
    plt.savefig(CURVES_PATH)
    
    if best_state is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in best_state.items()}
        )
