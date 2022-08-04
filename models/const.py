import torch

DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")