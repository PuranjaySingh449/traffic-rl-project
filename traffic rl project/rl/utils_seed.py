import random
import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility across random, numpy, and PyTorch.
    
    Args:
        seed (int): Random seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
