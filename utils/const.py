import torch

DEFAULT_DTYPE = torch.float
DEFAULT_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOSS_BLOW_UP_THRESHOLD = 1e32
ARXIV_MODEL_LATENT_VECTOR_SIZE = 6
MODES = ('train', 'valid', 'test')