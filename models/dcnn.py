import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from ..const import DEFAULT_DTYPE, DEFAULT_DEVICE

class DCNN(nn.Module):
    "Deep Convolutional Neural Network (DCNN)."
    
    def __init__(
        self, 
        input_channel: int,
        output_channel: int,
        hidden_channels: Optional[List[int]],
        kernel_sizes: List[Union[int, Tuple[int]]],
        conv_transpose: bool = False,
        strides: Union[List[int], int] = 1,
        paddings: Union[List[int], int] = 0, 
        dilations: Union[List[int], int] = 1, 
        groups: Union[List[int], int] = 1, 
        biases: Union[List[bool], bool] = True,
        padding_modes: Union[List[str], str] = 'zeros',
        leaky_relu_negative_slopes: Union[List[float], float] = 0.01,
        device: Optional[torch.device] = DEFAULT_DEVICE,
        dtype: Optional[torch.dtype] = DEFAULT_DTYPE
    ):
        """Deep Convolutional Neural Network (DCNN).
        For CNN specifications, see https://pytorch.org/docs/stable/generated/torch.conv.html.

        :param input_channel: Number of channels in the input image
        :type input_channel: int
        :param output_channel: Number of channels produced by the network
        :type output_channel: int
        :param hidden_channels: Number of channels produced in the hidden layer of the network,
            no hidden layers if None or empty.
        :type hidden_channels: Optional[List[int]]
        :param kernel_sizes: Size or a list of sizes of the convolving kernel. 
            Assumed to be homogeneous if int.
        :type kernel_sizes: List[Union[int, Tuple[int]]]
        :param conv_transpose: If True, use ConvTranspose2d instead of Conv2d, defaults to False.
        :type conv_transpose: bool
        :param strides: Stride or or a list of strides of the convolution, defaults to 1. 
            Assumed to be homogeneous if int.
        :type strides: Union[List[int], int], optional
        :param paddings: Padding or a list of paddings added to all four sides of the input, defaults to 0.
            Assumed to be homogeneous if int,
        :type paddings: Union[List[int], int], optional
        :param dilations: Spacing or a list of spacings between kernel elements, defaults to 1.
        :type dilations: Union[List[int], int], optional
        :param groups: Number or a list of numbers of blocked connections 
            from input channels to output channels, defaults to 1.
        :type groups: Union[List[int], int], optional
        :param biases: If True, adds a learnable bias to the output, defaults to True
        :type biases: Union[List[bool], bool], optional
        :param padding_modes: Modes or a list of modes for padding, defaults to 'zeros'
        :type padding_modes: Union[List[str], str], optional
        :param leaky_relu_negative_slopes: `negative_slope` or a list of `negative_slope`s of
            the leaky_relu layers in the neural network.
        :type leaky_relu_negative_slopes: Union[List[float], float]
        :param device: Model's device, defaults to 'cuda' if available, otherwise 'cpu'
        :type device: Optional[torch.device], optional
        :param dtype: Model's data type, defaults to `torch.float`
        :type dtype: Optional[torch.dtype], optional
        """
        
        conv = conv if not nn.Conv2d else nn.Conv2d
        
        # type checks
        if not (isinstance(hidden_channels, list) or (hidden_channels is None)):
            raise ValueError(f'hidden_channels must be a list of int or. Found: {type(hidden_channels)}.')
        for param, name, expected_type in zip(
            (kernel_sizes, strides, paddings, dilations, groups, biases, biases, padding_modes),
            ('kernel_sizes', 'strides', 'paddings', 'dilations', 'groups', 'biases', 'biases', 'padding_modes'),
            ((int, tuple), int, int, int, int, int, bool, str),
        ):
            if isinstance(param, list):
                # (len(hidden_channels) - 1) + 2  
                # 1 for input layer and 1 for output layer
                expected_length = len(hidden_channels) + 1  
                if len(param) < expected_length:
                    raise ValueError(
                        f'Not enough specification in {name}. '
                        f'If {name} is a list of parameters, '
                        f'it needs to have a length len(hidden_channels)+1 (i.e. {expected_length}), '
                        'where the extra two parameters at the beginning and end'
                        'accounts for the input and output layers, respectively.'
                    )
            elif isinstance(param, expected_type):  # homogeneous within the network
                pass
            else:
                type_str = str(expected_type).split("'")[-2]
                raise TypeError(
                    f"{name} expected to be a '{type_str}' or a list of '{type_str}'. "
                    f"Found: {type(param)}"
                )
          
        super().__init__()
        # member variables
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_channels = hidden_channels if (hidden_channels is not None) else []
        self.num_cnns = len(self.hidden_channels) + 1
        if isinstance(kernel_sizes, list):
            if len(kernel_sizes) == 1:
                # expand to the same length as hidden_channels
                # assuming homogeneity within the network
                kernel_sizes = [kernel_sizes[0]] * self.num_cnns
            self.kernel_sizes = kernel_sizes
        else:
            if isinstance(kernel_sizes, int):
                self.kernel_sizes = [kernel_sizes] * self.num_cnns
            elif isinstance(kernel_sizes, tuple):
                # must be a tuple of length 2
                if len(kernel_sizes) != 2:
                    raise ValueError(
                        'Each kernel size must be a tuple of 2 integers. '
                        f'Found: {len(kernel_sizes)=}.'
                    )
                self.kernel_sizes = [kernel_sizes] * self.num_cnns
            else:
                raise TypeError(
                    f'kernel_sizes expected to be a int or a tuple of two int. Found: {type(kernel_sizes)}.'
                )
        if isinstance(strides, list) and (len(strides) == 1):
            strides = strides[0]
        self.strides = strides if isinstance(strides, list) else [strides] * self.num_cnns
        if isinstance(paddings, list) and (len(paddings) == 1):
            paddings = paddings[0]
        self.paddings = paddings if isinstance(paddings, list) else [paddings] * self.num_cnns
        if isinstance(dilations, list) and (len(dilations) == 1):
            dilations = dilations[0]
        self.dilations = dilations if isinstance(dilations, list) else [dilations] * self.num_cnns
        if isinstance(groups, list) and (len(groups) == 1):
            groups = groups[0]
        self.groups = groups if isinstance(groups, list) else [groups] * self.num_cnns
        if isinstance(biases, list) and (len(biases) == 1):
            biases = biases[0]
        self.biases = biases if isinstance(biases, list) else [biases] * self.num_cnns
        if isinstance(padding_modes, list) and (len(padding_modes) == 1):
            padding_modes = padding_modes[0]
        self.padding_modes = padding_modes if isinstance(padding_modes, list) else [padding_modes] * self.num_cnns
        if isinstance(leaky_relu_negative_slopes, list) and (len(leaky_relu_negative_slopes) == 1):
            leaky_relu_negative_slopes = leaky_relu_negative_slopes[0]
        self.leaky_relu_negative_slopes = leaky_relu_negative_slopes if isinstance(
            leaky_relu_negative_slopes, list
        ) else [leaky_relu_negative_slopes] * self.num_cnns
        self.device = device
        self.dtype = dtype

        # neural networks
        # input -> output
        if len(self.hidden_channels) == 0:  
            self.cnn = nn.Sequential(
                conv(
                    in_channels=self.input_channel, 
                    out_channels=self.output_channel, 
                    kernel_size=self.kernel_sizes[0], 
                    stride=self.strides[0], 
                    padding=self.paddings[0], 
                    dilation=self.dilations[0], 
                    groups=self.groups[0], 
                    bias=self.biases[0], 
                    padding_mode=self.padding_modes[0], 
                    device=self.device, 
                    dtype=self.dtype
                ),
                nn.LeakyReLU(
                    negative_slope=self.leaky_relu_negative_slopes[0],
                    inplace=False
                ).to(self.device, self.dtype)
            )
        
        # input -> hidden layers -> output
        else:  
            # input -> hidden layers
            input_layer = nn.Sequential(
                conv(
                    in_channels = self.input_channel, 
                    out_channels = self.hidden_channels[0], 
                    kernel_size = self.kernel_sizes[0], 
                    stride = self.strides[0], 
                    padding = self.paddings[0], 
                    dilation = self.dilations[0], 
                    groups = self.groups[0], 
                    bias = self.biases[0], 
                    padding_mode = self.padding_modes[0], 
                    device = self.device, 
                    dtype = self.dtype
                ),
                nn.LeakyReLU(
                    negative_slope=self.leaky_relu_negative_slopes[0],
                    inplace=False
                ).to(self.device, self.dtype)
            )
            # hidden layers
            hidden_layers = nn.ModuleList()
            for i in range(self.num_cnns - 2):
                hidden_layer = nn.Sequential(
                    conv(
                        in_channels = self.hidden_channels[i], 
                        out_channels = self.hidden_channels[i+1], 
                        kernel_size = self.kernel_sizes[i+1], 
                        stride = self.strides[i+1], 
                        padding = self.paddings[i+1], 
                        dilation = self.dilations[i+1], 
                        groups = self.groups[i+1], 
                        bias = self.biases[i+1], 
                        padding_mode = self.padding_modes[i+1], 
                        device = self.device, 
                        dtype = self.dtype
                    ),
                    nn.LeakyReLU(
                        negative_slope=self.leaky_relu_negative_slopes[i+1],
                        inplace=False
                    ).to(self.device, self.dtype)
                )
                hidden_layers.append(hidden_layer)
            # last hidden layer -> output
            output_layer = nn.Sequential(
                conv(
                    in_channels=self.hidden_channels[-1],
                    out_channels=self.output_channel,
                    kernel_size=self.kernel_sizes[-1],
                    stride=self.strides[-1],
                    padding=self.paddings[-1],
                    dilation=self.dilations[-1],
                    groups=self.groups[-1],
                    bias=self.biases[-1],
                    padding_mode=self.padding_modes[-1],
                    device=self.device,
                    dtype=self.dtype
                ),
                nn.LeakyReLU(
                    negative_slope=self.leaky_relu_negative_slopes[-1],
                    inplace=False
                ).to(self.device, self.dtype)
            )
               
            # compose the NN 
            self.cnn = nn.Sequential(input_layer, *hidden_layers, output_layer)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of DCNN.

        :param x: Input tensor of shape (batch_size, n_channels, height, width)
            or (batch_size, height, width), which by assumption has 1 channel.
        :type x: torch.Tensor
        :raises TypeError: if x is not a torch.Tensor.
        :raises ValueError: when x is not a 3- or 4-dimensional tensor.
        :return: Output of the NN.
        :rtype: torch.Tensor
        """    
        # type check     
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x is expected to be a torch.Tensor. Found: {type(x)}")
       
        # send to device and dtype
        x = x.to(self.device, self.dtype)
        
        # dimension checks
        if x.dim() == 3:  # 1 channel by assumption
            if self.input_channel != 1:
                raise ValueError(
                    "Input x is assumed to have 1 channel. "
                    f"Found: {self.input_channel[1]=}"
                )
            x = x.unsqueeze(1)  # unsqueeze to 4 dimensional vectors
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(
                'Input tensor x must be 3 or 4 dimensional. '
                f'Found: {x.dim()=}'
            )
        return self.cnn(x)    