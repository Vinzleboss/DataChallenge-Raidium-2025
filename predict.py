import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import TEST_IMAGES_DIR, MODEL_PATH, OUTPUTS_DIR, PREDICTION_CSV, VERIF_FIGURE, device
from model import UnetCNN
from data import load_dataset

data_test = load_dataset(TEST_IMAGES_DIR)

@torch.no_grad()
def predict_test(model, images, device, batch_size=16):
    model.eval()
    preds = []
    for start in tqdm(range(0, len(images), batch_size), desc="Prédiction test"):
        x = torch.from_numpy(images[start:start + batch_size]).float().unsqueeze(1) / 255.0
        logits = model(x.to(device))
        preds.append(logits.argmax(dim=1).to(torch.uint8).cpu())
    return torch.cat(preds).numpy()                     # (500, 256, 256)

# --- Recharger le meilleur modèle
model = UnetCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
# --- Prédire les 500 images de test
pred_test = predict_test(model, data_test, device)
print("Prédictions :", pred_test.shape, pred_test.dtype, "| classes :", np.unique(pred_test)[:10], "...")

# --- Écrire le CSV au format de l'énoncé
submission = pd.DataFrame(pred_test.reshape(pred_test.shape[0], -1)).T
submission.index = [f"Pixel {i}" for i in range(submission.shape[0])]
submission.columns = [f"{k}.png" for k in range(submission.shape[1])]
OUTPUTS_DIR.mkdir(exist_ok=True)
submission.to_csv(PREDICTION_CSV)
print("CSV :", submission.shape)                          # attendu : (65536, 500)

# --- Relecture : recharger le CSV et comparer à l'image
check = pd.read_csv(PREDICTION_CSV, index_col=0, header=0).T
k = 0
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(data_test[k], cmap="gray")
axes[1].imshow(data_test[k], cmap="gray")
seg = check.iloc[k].values.reshape(256, 256)
axes[1].imshow(np.ma.masked_where(seg == 0, seg), cmap="tab20")
for ax in axes:
    ax.axis("off")
plt.savefig(VERIF_FIGURE)