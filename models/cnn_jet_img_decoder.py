from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from ..const import DEFAULT_DEVICE, DEFAULT_DTYPE
from .dcnn import DCNN
from .linear2d import Linear2D

class CNNJetImgDecoder(nn.Module):
    """CNN decoder for jet images that decodes vectors of size 
    `(batch_size, latent_vector_size)`
    in the latent space to jet images of size 
    `(batch_size, output_width, output_height)`
    
    Adapted from https://github.com/eugeniaring/Medium-Articles/blob/main/Pytorch/convAE.ipynb.
    """
    def __init__(
        self,
        latent_vector_size: int,
        output_height: int,
        output_width: int,
        unflatten_img_size: Union[Tuple[int, int], int],
        cnn_channels: List[int],
        cnn_kernel_sizes: List[Union[int, Tuple[int]]],
        unflatten_leaky_relu_negative_slope: Union[List[float], float] = 0.01,
        unflatten_hidden_widths: Optional[List[int]] = None,
        cnn_strides: Union[List[int], int] = 1,
        cnn_paddings: Union[List[int], int] = 0,
        cnn_dilations: Union[List[int], int] = 1,
        cnn_groups: Union[List[int], int] = 1,
        cnn_biases: Union[List[bool], bool] = True,
        cnn_padding_modes: Union[List[str], str] = 'zeros',
        cnn_leaky_relu_negative_slopes: Union[List[float], float] = 0.01,
        output_leaky_relu_negative_slope = 0.01,
        device: Optional[torch.device] = DEFAULT_DEVICE,
        dtype: Optional[torch.dtype] = DEFAULT_DTYPE
    ):
        """CNN decoder for jet images that decodes vectors of size 
        `(batch_size, latent_vector_size)`
        in the latent space to jet images of size 
        `(batch_size, output_width, output_height)`.

        :param latent_vector_size: Size of vector in the latent space.
        :type latent_vector_size: int
        :param output_height: Height of the output jet images.
        :type output_height: int
        :param output_width: Width of the output jet images.
        :type output_width: int
        :param unflatten_img_size: Jet image size that 
            the unflatten layers unflatten to.
        :type unflatten_img_size: Union[Tuple[int, int], int]
        :type cnn_channels: List[int]
        :param cnn_kernel_sizes: Kernel sizes (`kernel_sizes`) 
            in the DCNN model.
        :type cnn_kernel_sizes: List[Union[int, Tuple[int]]]
        :param cnn_strides: Strides (`strides`) in the DCNN model, 
            defaults to 1
        :type cnn_strides: Union[List[int], int], optional
        :param cnn_paddings: Paddings (`paddings`) in the DCNN model, 
            defaults to 0
        :type cnn_paddings: Union[List[int], int], optional
        :param cnn_dilations: Dilation (`dilations`) in the DCNN model, 
            defaults to 1
        :type cnn_dilations: Union[List[int], int], optional
        :param cnn_groups: Groups (`groups`) in the DCNN model, 
            defaults to 1
        :type cnn_groups: Union[List[int], int], optional
        :param cnn_biases: Biases (`biases`) in the DCNN model, 
            defaults to True
        :type cnn_biases: Union[List[bool], bool], optional
        :param cnn_padding_modes: Padding modes (`padding_modes`) 
            in the DCNN model, defaults to 'zeros'
        :type cnn_padding_modes: Union[List[str], str], optional
        :param cnn_leaky_relu_negative_slopes: Negative slopes of the leaky relu layers 
            in the CNNs, defaults to 0.01
        :type cnn_leaky_relu_negative_slopes: float, optional
        :param output_leaky_relu_negative_slope: `leaky_relu_negative_slopes`
            in the final output layer, defaults to 0.01
        :type output_leaky_relu_negative_slope: float, optional
        :param device: Model's device, defaults to gpu if available, otherwise cpu.
        :type device: Optional[torch.device], optional
        :param dtype: Model's data type, defaults to `torch.float`
        :type dtype: Optional[torch.dtype], optional
        """        
        super().__init__()
        
        # member variables
        self.output_height = output_height
        self.output_width = output_width
        self.latent_vector_size = latent_vector_size
        
        if (cnn_channels is None) or (len(cnn_channels) <= 0):
            raise ValueError("cnn_channels must be a non-empty list of integers.")
        self.cnn_params = {
            "cnn_channels": cnn_channels,
            "cnn_kernel_sizes": cnn_kernel_sizes,
            "cnn_strides": cnn_strides,
            "cnn_paddings": cnn_paddings,
            "cnn_dilations": cnn_dilations,
            "cnn_groups": cnn_groups,
            "cnn_biases": cnn_biases,
            "cnn_padding_modes": cnn_padding_modes,
            "cnn_leaky_relu_negative_slopes": cnn_leaky_relu_negative_slopes
        }
        
        # dimension check: unflatten_img_size needs to have the format (height, width)
        err_msg = 'unflatten_img_size must be a tuple of 2 integer, (height, width).'
        if isinstance(unflatten_img_size, torch.Tensor):
            if (unflatten_img_size.dim() != 1) or (unflatten_img_size.shape[0] != 2):
                 raise ValueError(err_msg)
            else:
                unflatten_img_size = tuple(unflatten_img_size)
        elif isinstance(unflatten_img_size, tuple) or isinstance(unflatten_img_size, list):
            if len(unflatten_img_size) == 1:  # assumed to be square
                unflatten_img_size = (unflatten_img_size, unflatten_img_size)
            elif len(unflatten_img_size) == 2:
                pass
            else:
                raise ValueError(err_msg)
        elif isinstance(unflatten_img_size, int):
            unflatten_img_size = (unflatten_img_size, unflatten_img_size)
        else:
            raise TypeError(err_msg)
        
        self.flatten_params = {
            "unflatten_img_size": unflatten_img_size,
            "unflatten_leaky_relu_negative_slope": unflatten_leaky_relu_negative_slope,
            "unflatten_hidden_widths": unflatten_hidden_widths
        }
        self.device = device if (device is not None) else DEFAULT_DEVICE
        self.dtype = dtype if (dtype is not None) else DEFAULT_DTYPE
    
        # unflatten layers: latent vector (-> ... )-> image
        linear_layers = nn.ModuleList()
        linear_layers_out_dim = cnn_channels[0] * unflatten_img_size[0] * unflatten_img_size[1]
        if (unflatten_hidden_widths is None) or (len(unflatten_hidden_widths) == 0):
            # no hidden layers: latent vector -> image
            linear_layers.append(nn.Linear(self.latent_vector_size, linear_layers_out_dim))
        else:
            # input -> hidden layers
            linear_layers.append(nn.Linear(self.latent_vector_size, unflatten_hidden_widths[0]))
            # hidden layers
            for i in range(len(unflatten_hidden_widths) - 1):
                linear_layers.append(nn.Linear(unflatten_hidden_widths[i], unflatten_hidden_widths[i+1]))
            # hidden layers -> output
            linear_layers.append(nn.Linear(unflatten_hidden_widths[-1], linear_layers_out_dim))
        
        unflatten_size = (cnn_channels[0], unflatten_img_size[0], unflatten_img_size[1])
        self.unflatten = nn.Sequential(
            *linear_layers,
            nn.LeakyReLU(negative_slope=unflatten_leaky_relu_negative_slope, inplace=False),
            nn.Unflatten(dim=1, unflattened_size=unflatten_size)
        ).to(self.device, self.dtype)
        
        # number of learnable parameters
        self.__num_param = sum(p.nelement() for p in self.parameters() if p.requires_grad)
        
        self.dcnn = DCNN(
            input_channel=cnn_channels[0],
            output_channel=1,
            hidden_channels=self.cnn_params["cnn_channels"][1:],
            kernel_sizes=self.cnn_params["cnn_kernel_sizes"],
            conv_transpose=True,
            strides=self.cnn_params["cnn_strides"],
            paddings=self.cnn_params["cnn_paddings"],
            dilations=self.cnn_params["cnn_dilations"],
            groups=self.cnn_params["cnn_groups"],
            biases=self.cnn_params["cnn_biases"],
            padding_modes=self.cnn_params["cnn_padding_modes"],
            leaky_relu_negative_slopes=self.cnn_params["cnn_leaky_relu_negative_slopes"],
            device=self.device,
            dtype=self.dtype
        )
        
        # quick hack to know the output dimension of the DCNN
        test_input = torch.rand(
            1, cnn_channels[0], *unflatten_img_size,
            device=self.device, dtype=self.dtype
        )
        test_output = self.dcnn(test_input)
        self.dcnn_out_img_size = tuple(test_output.shape[1:])
        
        # output layer
        self.output_layer = nn.Sequential(
            Linear2D(
                in_height=self.dcnn_out_img_size[-2], 
                in_width=self.dcnn_out_img_size[-1], 
                out_height=self.output_height, 
                out_width=self.output_width,
                device=self.device, dtype=self.dtype
            ),
            nn.LeakyReLU(
                negative_slope=output_leaky_relu_negative_slope,
                inplace=False
            )
        ).to(self.device, self.dtype)
        
           
    def l1_norm(self):
        """L1 norm of the model parameters."""        
        return sum(p.abs().sum() for p in self.parameters())
    
    def l2_norm(self):
        """L2 norm of the model parameters."""
        return sum(torch.pow(p, 2).sum() for p in self.parameters())
        
        
    def forward(self, latent_vec: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param latent_vec: vector in the latent space,
            with shape (batch_size, latent_vector_size)
        :type latent_vec: torch.Tensor
        :return: decoded images with size 
            (batch_size, output_channel, output_height, output_width)
        :rtype: torch.Tensor
        """
        # dimension check
        if latent_vec.dim() != 2:  # (batch_size, latent_vector_size)
            raise ValueError(
                'latent_vec must be a 1-dimensional vector. '
                f'Found: {latent_vec.shape=}'
            )
            
        # send to device and dtype
        latent_vec = latent_vec.to(self.device, self.dtype)
        
        latent_vec = self.unflatten(latent_vec)
        img = self.dcnn(latent_vec)
        img = self.output_layer(img)
        
        # (batch_size, 1, height, width) -> (batch_size, height, width)
        return img.squeeze(1)
    
    @property
    def num_learnable_parameters(self):
        return self.__num_param
