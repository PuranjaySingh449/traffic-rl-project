import os
import random
import numpy as np
import torch

def set_all_seeds(seed: int):
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN (if using GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[SEED] Global seed set to {seed}")
