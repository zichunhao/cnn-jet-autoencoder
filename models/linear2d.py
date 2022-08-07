from typing import Optional
import torch
from torch import nn
from ..const import DEFAULT_DEVICE, DEFAULT_DTYPE

class Linear2D(nn.Module):
    """Linear network for 2D feature: 
    (..., in_height, in_width) -> (..., out_height, out_width)
    """
    
    def __init__(
        self, 
        in_height: int,
        in_width: int,
        out_height: int,
        out_width: int,
        device: Optional[torch.device] = DEFAULT_DEVICE,
        dtype: Optional[torch.dtype] = DEFAULT_DTYPE
    ):
        """Linear network for 2D feature: 
        (..., in_height, in_width) -> (..., out_height, out_width)

        :param in_height: Input height.
        :type in_height: int
        :param in_width: Input width.
        :type in_width: int
        :param out_height: Output height.
        :type out_height: int
        :param out_width: Output width.
        :type out_width: int
        """        
        super().__init__()
        self.device = device if (device is not None) else DEFAULT_DEVICE
        self.dtype = dtype if (dtype is not None) else DEFAULT_DTYPE
        
        self.layer_height = nn.Linear(
            in_height, out_height,
            device=self.device, dtype=self.dtype
        )
        self.layer_width = nn.Linear(
            in_width, out_width,
            device=self.device, dtype=self.dtype
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Linear2D network: 
        (..., in_height, in_width) -> (..., out_height, out_width).

        :param x: input feature.
        :type x: torch.Tensor
        :return: output feature from the model.
        :rtype: torch.Tensor
        """
        if x.dim() < 2:
            raise ValueError(
                'x is expected to have dimension (..., in_height, in_width). '
                f'Found: {x.shape=}.'
            )  
        x = x.to(self.device, self.dtype)
        x = self.layer_height(x)
        x = x.transpose(-1, -2)
        x = self.layer_width(x)
        return x.transpose(-1, -2)
        
    