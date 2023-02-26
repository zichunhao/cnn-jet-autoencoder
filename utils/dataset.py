import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from utils.const import DEFAULT_DEVICE, DEFAULT_DTYPE


class JetImageDataset(Dataset):
    def __init__(
        self,
        jet_imgs: torch.Tensor,
        normalize: Optional[str] = None,
        shuffle: bool = True,
        device: Optional[torch.device] = DEFAULT_DEVICE,
        dtype: Optional[torch.dtype] = DEFAULT_DTYPE,
    ):
        """Data loader for jet images dataset.

        :param jet_imgs: Jet images to load.
        :type jet_imgs: torch.Tensor
        :param normalize: Whether to normalize the data, defaults to True.
        :type normalize: bool, optional
        :param shuffle: Whether to shuffle the data, defaults to True.
        :type shuffle: bool, optional
        :param device: Device to load the data to,
            defaults to 'cuda' if available, otherwise 'cpu'.
        :type device: Optional[torch.device], optional
        :param dtype: Data type to load the data to, defaults to 'torch.float32'.
        :type dtype: Optional[torch.dtype], optional
        """
        self.device = device if (device is not None) else DEFAULT_DEVICE
        self.dtype = dtype if (dtype is not None) else DEFAULT_DTYPE
        jet_imgs = jet_imgs.to(self.device, self.dtype)

        # normalize jet images by dividing the max value in each jet image
        if normalize is not None:
            if normalize.lower() == "max":
                # normalize by the max value in each jet image
                self.norm_factors = (
                    jet_imgs.abs()
                    .max(dim=-1, keepdim=True)
                    .values.max(dim=-2, keepdim=True)
                    .values
                )
            elif normalize.lower() == "sum":
                # normalize by the sum of all values in each jet image
                self.norm_factors = jet_imgs.sum(dim=-1, keepdim=True).sum(
                    dim=-2, keepdim=True
                )
            elif normalize.lower() in ("", "none"):
                # no normalization
                self.norm_factors = torch.ones(
                    (len(jet_imgs), 1, 1), device=self.device, dtype=self.dtype
                )
            else:
                logging.error(
                    f"Unknown normalization method: {normalize}. Use 'sum' instead."
                )
                self.norm_factors = jet_imgs.sum(dim=-1, keepdim=True).sum(
                    dim=-2, keepdim=True
                )
            self.jet_imgs = jet_imgs / self.norm_factors
        else:
            self.norm_factors = torch.ones(
                (len(jet_imgs), 1, 1), device=self.device, dtype=self.dtype
            )
            self.jet_imgs = jet_imgs

        self.shuffle = shuffle
        if self.shuffle:
            self.perm = torch.randperm(len(self.jet_imgs))
        else:
            self.perm = None

    def __len__(self) -> int:
        return len(self.jet_imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from the dataset.

        :param idx: Index of the jet image of interest.
        :type idx: int
        :return: (jet_img, norm_factor) at the given index `idx`.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if self.shuffle:
            idx = self.perm[idx]
        return (self.jet_imgs[idx], self.norm_factors[idx])
