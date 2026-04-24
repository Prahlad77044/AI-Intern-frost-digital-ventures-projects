import torch
import random
import numpy as np

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Categories
CATEGORY_MAP = ["Rights", "Governance", "Judiciary", "Federalism", "Other"]
cat2idx = {c: i for i, c in enumerate(CATEGORY_MAP)}
idx2cat = {i: c for c, i in cat2idx.items()}

# Training hyperparameters
HIDDEN_DIM = 256
LR = 0.005
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 20

EMBED_MODEL = "all-mpnet-base-v2"
