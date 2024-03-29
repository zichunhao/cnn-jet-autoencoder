import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from functools import reduce
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from models.dcnn import DCNN
from utils import DEFAULT_DEVICE, DEFAULT_DTYPE


class CNNJetImgEncoder(nn.Module):
    """CNN autoencoder for jet images that
    encodes a jet image of shape  `(batch_size, input_height, input_width)`
    to a 1-dimensional vector of shape `(batch_size, latent_vector_size)`.

    Adapted from https://github.com/eugeniaring/Medium-Articles/blob/main/Pytorch/convAE.ipynb.
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        latent_vector_size: int,
        cnn_channels: List[int],
        cnn_kernel_sizes: List[Union[int, Tuple[int]]],
        cnn_strides: Union[List[int], int] = 1,
        cnn_paddings: Union[List[int], int] = 0,
        cnn_dilations: Union[List[int], int] = 1,
        cnn_groups: Union[List[int], int] = 1,
        cnn_biases: Union[List[bool], bool] = True,
        cnn_padding_modes: Union[List[str], str] = "zeros",
        cnn_leaky_relu_negative_slopes: Union[List[float], float] = 0.01,
        cnn_use_intermediates: bool = False,
        flatten_leaky_relu_negative_slope: float = 0.01,
        flatten_hidden_widths: Optional[List[int]] = None,
        device: Optional[torch.device] = DEFAULT_DEVICE,
        dtype: Optional[torch.dtype] = DEFAULT_DTYPE,
    ):
        """CNN autoencoder for jet images that
        encodes a jet image of shape  `(batch_size, input_height, input_width)`
        to a 1-dimensional vector of shape `(batch_size, latent_vector_size)`.

        :param input_height: height of input image.
        :type input_height: int
        :param input_width: width of input image.
        :type input_width: int
        :param latent_vector_size: size of latent vector.
        :type latent_vector_size: int
        :param cnn_channels: Channels in the DCNN model.
        :type cnn_channels: List[int]
        :param cnn_kernel_sizes: Kernel sizes (`kernel_sizes`)
        in the DCNN model.
        :type cnn_kernel_sizes: List[Union[int, Tuple[int]]]
        :param cnn_strides: Strides (`strides`) in the DCNN model,
        defaults to 1.
        :type cnn_strides: Union[List[int], int], optional
        :param cnn_paddings: Paddings (`paddings`) in the DCNN model,
        defaults to 0.
        :type cnn_paddings: Union[List[int], int], optional
        :param cnn_dilations: Dilation (`dilations`) in the DCNN model,
        defaults to 1.
        :type cnn_dilations: Union[List[int], int], optional
        :param cnn_groups: Groups (`groups`) in the DCNN model,
        defaults to 1.
        :type cnn_groups: Union[List[int], int], optional
        :param cnn_biases: Biases (`biases`) in the DCNN model,
        defaults to True.
        :type cnn_biases: Union[List[bool], bool], optional
        :param cnn_padding_modes: Padding modes (`padding_modes`)
        in the DCNN model, defaults to 'zeros'.
        :type cnn_padding_modes: Union[List[str], str], optional
        :param cnn_leaky_relu_negative_slopes: Negative slopes of the leaky relu layers
        in the CNNs, defaults to 0.01.
        :type cnn_leaky_relu_negative_slopes: float, optional
        :param cnn_use_intermediates: Whether to keep track intermediate maps of CNNs,
        defaults to False.
        :type cnn_use_intermediates: bool, optional
        :param flatten_leaky_relu_negative_slope: `leaky_relu_negative_slopes`
        in the DCNN model, defaults to 0.01.
        :type flatten_leaky_relu_negative_slope: Union[List[float], float], optional
        :param flatten_hidden_widths: widths of hidden linear layers in the flatten layers,
        defaults to None. When `flatten_hidden_widths` is None or an empty list,
        there is no hidden layer.
        :type flatten_hidden_widths: Optional[List[int]], optional
        :param device: Model's device, defaults to gpu if available, otherwise cpu.
        :type device: Optional[torch.device], optional
        :param dtype: Model's data type, defaults to `torch.float`.
        :type dtype: Optional[torch.dtype], optional
        """
        super(CNNJetImgEncoder, self).__init__()

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
            "cnn_leaky_relu_negative_slopes": cnn_leaky_relu_negative_slopes,
            "cnn_use_intermediates": cnn_use_intermediates,
        }

        self.flatten_params = {
            "flatten_leaky_relu_negative_slope": flatten_leaky_relu_negative_slope,
            "flatten_hidden_widths": flatten_hidden_widths,
        }

        self.device = device if (device is not None) else DEFAULT_DEVICE
        self.dtype = dtype if (dtype is not None) else DEFAULT_DTYPE

        # deep CNN
        self.dcnn = DCNN(
            input_channel=1,
            output_channel=self.cnn_params["cnn_channels"][-1],
            hidden_channels=self.cnn_params["cnn_channels"][:-1],
            kernel_sizes=self.cnn_params["cnn_kernel_sizes"],
            conv_transpose=False,
            strides=self.cnn_params["cnn_strides"],
            paddings=self.cnn_params["cnn_paddings"],
            dilations=self.cnn_params["cnn_dilations"],
            groups=self.cnn_params["cnn_groups"],
            biases=self.cnn_params["cnn_biases"],
            padding_modes=self.cnn_params["cnn_padding_modes"],
            leaky_relu_negative_slopes=self.cnn_params[
                "cnn_leaky_relu_negative_slopes"
            ],
            device=self.device,
            dtype=self.dtype,
            keep_intermediates=cnn_use_intermediates,
        )

        test_input = torch.rand(
            1, 1, input_height, input_width, device=self.device, dtype=self.dtype
        )
        if cnn_use_intermediates:
            test_output, test_feature_maps = self.dcnn(test_input)
            intermediate_entries = 0
            for feature_map in test_feature_maps:
                # to be concatenated
                intermediate_dim = tuple(feature_map.shape[1:])
                intermediate_entries += reduce(lambda x, y: x * y, intermediate_dim)
        else:
            test_output = self.dcnn(test_input)
            intermediate_entries = 0
        self.dcnn_out_img_size = tuple(test_output.shape[1:])
        output_num_entries = (
            reduce(lambda x, y: x * y, self.dcnn_out_img_size) + intermediate_entries
        )

        # flatten layer: 3d feature (2d image with multiple channels) -> 1d vector
        linear_layers = nn.ModuleList()
        flatten_hidden_widths = self.flatten_params["flatten_hidden_widths"]
        if (flatten_hidden_widths is None) or (len(flatten_hidden_widths) == 0):
            # no hidden layers: input -> output
            linear_layers.append(nn.Linear(output_num_entries, latent_vector_size))
        else:
            # hidden layers: input -> hidden -> output
            # input -> hidden
            linear_layers.append(
                nn.Linear(output_num_entries, flatten_hidden_widths[0])
            )
            # hidden layers
            for i in range(len(flatten_hidden_widths) - 1):
                linear_layers.append(
                    nn.Linear(flatten_hidden_widths[i], flatten_hidden_widths[i + 1])
                )
            # hidden -> output
            linear_layers.append(
                nn.Linear(flatten_hidden_widths[-1], latent_vector_size)
            )
        linear_layers = linear_layers.to(self.device, dtype=self.dtype)

        self.flatten = nn.Sequential(
            nn.Flatten(start_dim=1),
            *linear_layers,
            nn.LeakyReLU(
                self.flatten_params["flatten_leaky_relu_negative_slope"], inplace=False
            ),
        ).to(self.device, self.dtype)

        # number of learnable parameters
        self.__num_param = sum(
            p.nelement() for p in self.parameters() if p.requires_grad
        )

    def l1_norm(self):
        """L1 norm of the model parameters."""
        return sum(p.abs().sum() for p in self.parameters())

    def l2_norm(self):
        """L2 norm of the model parameters."""
        return sum(torch.pow(p, 2).sum() for p in self.parameters())

    def forward(self, jet_img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNNJetEncoder model.

        :param jet_img: jet images with shape (batch_size, input_height, input_width)
            or (batch_size, n_channels, input_height, input_width).
        :type jet_img: torch.Tensor
        :raises ValueError: if `jet_img` has an invalid shape.
        :return: Encoded feature with shape (batch_size, latent_vector_size).
        :rtype: torch.Tensor
        """
        # dimension checks
        if jet_img.dim() == 3:  # 1 channel by assumption
            jet_img = jet_img.unsqueeze(1)  # unsqueeze to 4 dimensional vectors
        elif jet_img.dim() == 4:
            pass
        else:
            raise ValueError(
                "Input tensor jet_img must be 3 or 4 dimensional. "
                f"Found: {jet_img.dim()=}"
            )

        # send to device and dtype
        jet_img = jet_img.to(self.device, self.dtype)
        if not self.cnn_params["cnn_use_intermediates"]:
            return self.flatten(self.dcnn(jet_img))
        else:
            output, feature_maps = self.dcnn(jet_img)
            output = output.flatten(start_dim=1)
            return self.flatten(torch.cat([output, *feature_maps], dim=1))

    @property
    def num_learnable_parameters(self):
        return self.__num_param
