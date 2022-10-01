from typing import Union
import torch
from torch import nn

EPS = 1e-16


class CustomMSELoss(nn.Module):
    def __init__(self, lambda_nz: Union[int, float]):
        """Custom MSE that penalize nonzero values in cells that are supposed to be 0:
        :math:`\mathcal{L} = \sum_{t_i \neq 0} (t_i - x_i)^2 + \lambda \sum_{t_i = 0} (t_i - x_i)^2`,
        where :math:`t_i` is the :math:`i`-th target value and :math:`x_i` is the :math:`i`-th input value.
        
        :param lambda_nz: Penalization score for nonzero values in cells that are supposed to be 0.
        :type lambda_nz: int or float
        """
        super(CustomMSELoss, self).__init__()
        self.lambda_nz = lambda_nz
        self.mse = nn.MSELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lambda_nz == 1:
            return self.mse(input, target)
        zero_cells = (target == 0)
        loss_nonzero = self.mse(input*(~zero_cells), target*(~zero_cells))
        if self.lambda_nz == 0:
            return loss_nonzero
        loss_zero = self.mse(input*zero_cells, target*zero_cells)
        return loss_nonzero + self.lambda_nz*loss_zero


class RelativeMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        # NOTE: zero cells are problematic
        return torch.mean(torch.square((input - target) / (target + EPS)))
