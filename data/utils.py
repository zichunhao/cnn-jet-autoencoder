import argparse
import torch
import logging

DEFAULT_DTYPE = torch.float32

def get_dtype(arg: argparse.Namespace) -> torch.dtype:
    """Parse torch.dtype from input string"""

    if arg.lower() in ('float32', 'float64'):
        return torch.float
    elif arg.lower() in ('double', 'float64'):
        return torch.float64
    else:
        logging.warning(
            f"{arg} is not a recognizable device. "
            f"Defaulting to {DEFAULT_DTYPE}."
        )
        return DEFAULT_DTYPE