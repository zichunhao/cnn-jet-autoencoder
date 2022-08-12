import os
import torch

def mkdir(path: str, exist_ok: bool = True) -> str:
    """Make directory if it doesn't exist and return it.

    :param path: The path to the directory.
    :type path: str
    :return: path
    :rtype: str
    """    
    os.makedirs(path, exist_ok=exist_ok)
    return path

def get_compression_rate(
    img_height: int,
    img_width: int,
    latent_vector_size: int
) -> float:
    """Get the compression rate given
        - image height `img_height`, 
        - image width `img_width`, and 
        - vector size in the latent space `latent_vector_size`.

    :param img_height: Image height.
    :type img_height: int
    :param img_width: Image width.
    :type img_width: int
    :param latent_vector_size: Vector size in the latent space.
    :type latent_vector_size: int
    :return: Compression rate of the autoencoder.
    :rtype: float
    """    
    return (img_height * img_width) / latent_vector_size


def get_eps(dtype: torch.dtype) -> float:
    """Get epsilon values based on the data type.

    :param dtype: The data type.
    :type dtype: torch.dtype
    :return: 1e-16 if dtype is double, otherwise 1e-12
    :rtype: float
    """    
    if dtype == torch.float64:
        return 1e-16
    else:
        return 1e-12