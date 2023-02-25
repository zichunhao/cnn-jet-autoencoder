import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from models.linear2d import Linear2D


class CNNAE_UCSD_PHYS139_Encoder(nn.Module):
    def __init__(self, im_size: int = 40, latent_dim: int = 2) -> None:
        """
        Encoder of the CNN Autoencoder from the UCSD PHYS 139/239 course
        by Prof. Javier Duarte (PyTorch version).
        
        Source: https://github.com/jmduarte/phys139_239/blob/main/notebooks/07_Autoencoder.ipynb
        
        :param im_size: Size of the image to be encoded.
        :type im_size: int
        :param latent_dim: Dimension of the latent space.
        :type latent_dim: int
        """
        super(CNNAE_UCSD_PHYS139_Encoder, self).__init__()

        self.im_size = im_size
        self.latent_dim = latent_dim

        encoder_dim = (im_size // 2) // 2

        # input shape: (batch_size, 1, im_size, im_size)
        self.encoder = nn.Sequential(
            # x_in = Input(shape=(im_size, im_size, 1))
            # x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same")(x_in)
            nn.Conv2d(
                in_channels=1,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1) if im_size % 2 == 0 else (0, 0),
            ),
            nn.ReLU(),
            # # x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same")(x)
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1) if (im_size // 2 % 2) == 0 else (0, 0),
            ),
            nn.ReLU(),
            # # x = Flatten()(x)
            nn.Flatten(start_dim=1),  # keep batch dim
            nn.Linear(encoder_dim * encoder_dim * 128, latent_dim),
        )
        
        # number of learnable parameters
        self.__num_param = sum(
            p.nelement() for p in self.parameters() if p.requires_grad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of encoder.

        :param x: Image to be encoded.
        :type x: torch.Tensor
        :raises ValueError: When input tensor is not 40x40.
        :return: encoded tensor.
        :rtype: torch.Tensor
        """
        if len(x.shape) == 3:  # (bs, im_size, im_size)
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:  # (bs, 1, im_size, im_size)
            if x.shape[1] != 1:
                raise ValueError("Input tensor must have 1 channel.")
        else:
            raise ValueError(
                "Input tensor must have 3 or 4 dimensions, with 1 channel."
            )

        if x.shape[-1] != self.im_size or x.shape[-2] != self.im_size:
            raise ValueError(
                "Input image must have size im_size x im_size. "
                f"Got {x.shape[-1]}x{x.shape[-2]}."
            )
        return self.encoder(x)
    
    @property
    def num_learnable_parameters(self) -> int:
        """Number of learnable parameters.

        :return: Number of learnable parameters.
        :rtype: int
        """
        return self.__num_param
    
    def l1_norm(self):
        """L1 norm of the model parameters."""
        return sum(p.abs().sum() for p in self.parameters())

    def l2_norm(self):
        """L2 norm of the model parameters."""
        return sum(torch.pow(p, 2).sum() for p in self.parameters())


class CNNAE_UCSD_PHYS139_Decoder(nn.Module):
    def __init__(self, im_size: int = 40, latent_dim: int = 2) -> None:
        """
        Decoder of the CNN Autoencoder from the UCSD PHYS 139/239 course
        by Prof. Javier Duarte (PyTorch version).
        
        Source: https://github.com/jmduarte/phys139_239/blob/main/notebooks/07_Autoencoder.ipynb
        
        :param im_size: Size of the image to be encoded.
        :type im_size: int
        :param latent_dim: Dimension of the latent space.
        :type latent_dim: int
        """
        super(CNNAE_UCSD_PHYS139_Decoder, self).__init__()
        encoder_dim = (im_size // 2) // 2
        self.decoder = nn.Sequential(
            # x = Dense(int(im_size * im_size / 16) * 128, activation="relu")(x_enc)
            nn.Linear(latent_dim, encoder_dim * encoder_dim * 128),
            # x = Reshape((int(im_size / 4), int(im_size / 4), 128))(x)
            Reshape(-1, 128, encoder_dim, encoder_dim),
            # x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same")(x)
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1),
            ),
            nn.ZeroPad2d((0, 1, 0, 1)) if (im_size // 2) % 2 == 1 else nn.Identity(),
            # x = Conv2DTranspose(1, kernel_size=(3, 3), strides=(2, 2), activation="linear", padding="same")(x)
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=1,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1),
            ),
            nn.ZeroPad2d((0, 1, 0, 1)) if im_size % 2 == 1 else nn.Identity(),
            Linear2D(im_size, im_size, im_size, im_size),
            # x_out = Softmax(name="softmax", axis=[-2, -3])(x)
            nn.Softmax2d(),
        )
        
        # number of learnable parameters
        self.__num_param = sum(
            p.nelement() for p in self.parameters() if p.requires_grad
        )

    def forward(self, x: torch.Tensor, remove_channel: bool = True) -> torch.Tensor:
        """Forward pass of decoder.

        :param x: Encoded tensor.
        :type x: torch.Tensor
        :param remove_channel: Whether to remove the channel dimension, defaults to True
        :type remove_channel: bool, optional
        :raises ValueError: When input tensor is not 2D.
        :return: Decoded tensor.
        :rtype: torch.Tensor
        """
        if len(x.shape) != 2:
            raise ValueError("Input tensor must have 2 dimensions.")
        x_recons = self.decoder(x)
        if remove_channel:
            x_recons = x_recons.squeeze(1)
        return x_recons

    @property
    def num_learnable_parameters(self) -> int:
        """Number of learnable parameters.

        :return: Number of learnable parameters.
        :rtype: int
        """
        return self.__num_param
    
    def l1_norm(self):
        """L1 norm of the model parameters."""
        return sum(p.abs().sum() for p in self.parameters())

    def l2_norm(self):
        """L2 norm of the model parameters."""
        return sum(torch.pow(p, 2).sum() for p in self.parameters())

class Reshape(nn.Module):
    def __init__(self, *args) -> None:
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.shape)


class Conv2dSame(nn.Conv2d):
    """Conv2d with "same" padding.

    The output image shape is given by `(img_h//stride[0], img_w//stride[1])`

    The code structure is adapted from https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351
    but with reference to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """

    def __get_pad(
        self, kernel_size: int, padding: int, stride: int, dilation: int
    ) -> Tuple[int, int]:
        # define: in_dim_pad = in_dim + pad.
        # solve: in_dim / stride = out_dim(in_dim_pad),
        # where out_dim(x) = (x + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
        # as provided in https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        pad = 1 - dilation + dilation * kernel_size - 2 * padding - stride
        pad = max(pad, 0)
        pad_1 = pad // 2
        pad_2 = pad - pad_1
        return pad_1, pad_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        pad_top, pad_bottom = self.__get_pad(
            kernel_size=self.kernel_size[0],
            padding=self.padding[0],
            stride=self.stride[0],
            dilation=self.dilation[0],
        )
        pad_left, pad_right = self.__get_pad(
            kernel_size=self.kernel_size[1],
            padding=self.padding[1],
            stride=self.stride[1],
            dilation=self.dilation[1],
        )

        if (pad_top > 0) or (pad_bottom > 0) or (pad_left > 0) or (pad_right > 0):
            x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
