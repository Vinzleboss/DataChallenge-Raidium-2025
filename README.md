# DataChallenge-Raidium-2025
Datachallenge Raidium 2025 : U-Net for CT scans segmentation over a partially labelled training set

## Structure

```
├── config.py      # chemins, hyperparamètres, nom de l'expérience, device
├── model.py       # architecture UnetCNN
├── data.py        # load_dataset, CTDataset, plot_slice_seg
├── losses.py      # cross_entropy_modified
├── metrics.py     # scores DICE, evaluate_dice
├── train.py       # boucle d'entraînement + courbes de convergence
├── main.py        # script d'entraînement
├── predict.py     # prédiction sur le test + CSV de soumission
├── Tri_zones.py   # annotation manuelle des images
├── data/          # données du challenge (non versionnées, sauf annotated_labels.json)
├── models/        # poids des modèles (.pt)
├── figures/       # courbes et visualisations
├── outputs/       # CSV de prédiction (non versionnés)
└── notebooks/     # notebook d'exemple du challenge
```

## Installation

Avec conda (recommandé, utilise le GPU du Mac via MPS) :

```bash
conda env create -f environment.yml
conda activate datachall_radium_2026
```

Ou avec pip : `pip install -r requirements.txt`

Placer les données du challenge dans `data/` :

```
data/
├── train-images/
├── test-images/
├── y_train.csv
└── annotated_labels.json
```

## Utilisation

1. Choisir un nom d'expérience dans `config.py` (`EXPERIMENT_NAME`) et régler les hyperparamètres.
2. Entraîner : `python main.py` → `models/<EXPERIMENT_NAME>.pt` et `figures/<EXPERIMENT_NAME>_courbes.png`.
   L'entraînement refuse de démarrer si ce modèle existe déjà, pour ne pas l'écraser.
3. Prédire : `python predict.py` → `outputs/<EXPERIMENT_NAME>_y_test_pred.csv`.
