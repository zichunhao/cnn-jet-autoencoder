import torch

DEFAULT_DTYPE = torch.float
DEFAULT_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")