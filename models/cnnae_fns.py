from typing import Optional
import torch
from torch import nn

from utils.const import DEFAULT_DEVICE, DEFAULT_DTYPE

class CNNAEFNSEncoder(nn.Module):
    def __init__(
        self, 
        batch_norm: bool = True,
        device: Optional[torch.device] = DEFAULT_DEVICE,
        dtype: Optional[torch.dtype] = DEFAULT_DTYPE,
        *args, **kwargs
    ) -> None:
        """
        CNNAE given in "Searching for new physics with deep autoencoders"
        by Farina, Nakai, and Shih (FNS) (arXiv:1808.08992, 2020).
        """        
        super(CNNAEFNSEncoder, self).__init__()

        # input shape: (batch_size, 1, 40, 40)
        self.encoder = nn.Sequential(
            # layer = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(layer)
            nn.Conv2d(1, 128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            # layer = MaxPooling2D(pool_size=(2, 2),padding='same')(layer)
            nn.MaxPool2d(kernel_size=(2, 2)),
            # layer = Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same')(layer)
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.ReLU(),
            # layer = MaxPooling2D(pool_size=(2, 2),padding='same')(layer)
            nn.MaxPool2d(kernel_size=(2, 2)),
            # layer = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(layer)
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.ReLU(),
            # layer = Flatten()(layer)
            nn.Flatten(start_dim=1),
            # layer = Dense(32, activation='relu')(layer)
            nn.Linear(12800, 32),
            nn.ReLU(),
            # layer = Dense(6)(layer)
            nn.Linear(32, 6)
        ).to(device=device, dtype=dtype)
        
        self.__num_param = sum(p.nelement() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CNNAEFNSEncoder.

        :param x: Image to be encoded.
        :type x: torch.Tensor
        :raises ValueError: When input tensor is not 40x40.
        :return: encoded tensor.
        :rtype: torch.Tensor
        """        
        if x.shape[-1] != 40 or x.shape[-2] != 40:
            raise ValueError(
                "Input image must have size 40x40. "
                f"Got {x.shape[-1]}x{x.shape[-2]}."
            )
            
        if len(x.shape) == 3:
            # add channel dimension
            x = x.unsqueeze(1)
        
        return self.encoder(x)
    
    @property
    def num_learnable_parameters(self):
        return self.__num_param
    
    def l1_norm(self):
        """L1 norm of the model parameters."""
        return sum(p.abs().sum() for p in self.parameters())

    def l2_norm(self):
        """L2 norm of the model parameters."""
        return sum(torch.pow(p, 2).sum() for p in self.parameters())

class CNNAEFNSDecoder(nn.Module):
    def __init__(
            self, 
            batch_norm: bool = True,
            device: Optional[torch.device] = DEFAULT_DEVICE,
            dtype: Optional[torch.dtype] = DEFAULT_DTYPE, 
            *args, **kwargs
        ) -> None:
        super(CNNAEFNSDecoder, self).__init__()
        # encoded = layer
        self.decoder = nn.Sequential(
            # layer = Dense(32, activation='relu')(encoded)
            nn.Linear(6, 32),
            nn.ReLU(),
            # layer = Dense(12800, activation='relu')(layer)
            nn.Linear(32, 12800),
            nn.ReLU(),
            # layer = Reshape((10, 10, 128))(layer)
            Reshape(-1, 128, 10, 10),
            # layer = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(layer)
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.ReLU(),
            # layer = UpSampling2D((2, 2))(layer)
            nn.Upsample((20, 20)),  # layer.shape: (batch_size, 128, 10, 10)
            # layer = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(layer)
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.ReLU(),
            # layer = UpSampling2D((2, 2))(layer)
            nn.Upsample((40, 40)),  # layer.shape: (batch_size, 128, 20, 20)
            # layer = Conv2D(1, kernel_size=(3, 3), padding='same')(layer)
            nn.Conv2d(128, 1, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(1) if batch_norm else nn.Identity(),
            nn.ReLU(),
            # layer = Reshape((1, 1600))(layer)
            Reshape(-1, 1, 1600),
            # # layer = Activation('softmax')(layer)
            # nn.Softmax(dim=-1),
            # decoded = Reshape((40, 40, 1))(layer)
            Reshape(-1, 1, 40, 40)
        ).to(device=device, dtype=dtype)
        
        self.__num_param = sum(p.nelement() for p in self.parameters() if p.requires_grad)
        
    @property
    def num_learnable_parameters(self):
        return self.__num_param
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CNNAEFNSDecoder.

        :param x: Latent space vector to be decoded.
        :type x: torch.Tensor
        :return: Decoded image.
        :rtype: torch.Tensor
        """        
        return self.decoder(x).squeeze(1)  # remove channel dimension
    
    def l1_norm(self):
        """L1 norm of the model parameters."""
        return sum(p.abs().sum() for p in self.parameters())

    def l2_norm(self):
        """L2 norm of the model parameters."""
        return sum(torch.pow(p, 2).sum() for p in self.parameters())

class Reshape(nn.Module):  
    def __init__(self, *args) -> None:
        """Reshape module to be compatible with torch.nn.Sequential. 
        Initialized with the same arguments as torch.reshape.
        """        
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.shape)
    
    def __repr__(self) -> str:
        # ignore batch dimension
        return f'Reshape{self.shape[1:]}' 
