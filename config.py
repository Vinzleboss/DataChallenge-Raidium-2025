from pathlib import Path
import torch

# --- Chemins (relatifs à la racine du projet, marche aussi dans Docker)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_IMAGES_DIR = DATA_DIR / "train-images"
TEST_IMAGES_DIR = DATA_DIR / "test-images"
Y_TRAIN_CSV = DATA_DIR / "y_train.csv"
ANNOTATED_LABELS_JSON = DATA_DIR / "annotated_labels.json"

# --- Expérience : change ce nom avant chaque nouvel entraînement
EXPERIMENT_NAME = "Model_Caneaux_double"
MODEL_PATH = MODELS_DIR / f"{EXPERIMENT_NAME}.pt"
CURVES_PATH = FIGURES_DIR / f"{EXPERIMENT_NAME}_courbes.png"

# --- Prédiction
PREDICTION_CSV = OUTPUTS_DIR / f"{EXPERIMENT_NAME}_y_test_pred.csv"
VERIF_FIGURE = FIGURES_DIR / f"{EXPERIMENT_NAME}_verif_test.png"

# --- Données
NUM_CLASSES = 54
N_ANNOTATED = 800
TRAIN_VAL_SPLIT = [640, 160]

# --- Augmentation
AUG_DEGREES = 15
AUG_SCALE = (0.7, 1.4)

# --- Entraînement
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 16
SPLIT_SEED = 42
TORCH_SEED = 0

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
