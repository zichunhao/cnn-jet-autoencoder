import argparse
import logging
import torch
from math import inf
from .const import DEFAULT_DEVICE, DEFAULT_DTYPE

def parse_data_settings(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        '--data-paths', '-dp',
        nargs="+", type=str, metavar='',
        help='Paths to the training data (jet images).'
    )
    parser.add_argument(
        '--normalize',
        default=False, action='store_true',
        help='Whether to normalize the jet images.'
    )
    return parser

def parse_training_settings(
    parser: argparse.ArgumentParser,
    test: bool = False
) -> argparse.Namespace:
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float, metavar='', default=1e-3,
        help='Learning rate of training.'
    )
    parser.add_argument(
        '--batch-size', '-bs',
        type=int, metavar='', default=256,
        help='Batch size of training.'
    )
    parser.add_argument(
        '--optimizer', '-o',
        type=str, metavar='', default='adam',
        help='Optimizer to be used in the training. '
        'Supported choices: (adam, rmsprop)'
    )
    parser.add_argument(
        '--num-epochs', '-e',
        type=int, default=1000, metavar='',
        help='Number of epochs for training.'
    )
    parser.add_argument(
        '--lambda-nz', '-lnz',
        type=float, default=1e3, metavar='',
        help='Weights MSELoss of pixels that are supposed to be zero. '
        '(according to the target image).'
    )
    
    # training-validation split
    parser.add_argument(
        '--test-size',
        type=float, default=0.35, metavar='',
        help='If float, should be between 0.0 and 1.0 '
        'and represent the proportion of the dataset '
        'to include in the test split. '
        'If int, represents the absolute number of test samples. '
        'Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.'
    )
    
    # device and dtype
    parser.add_argument(
        '--device',
        type=get_dtype, metavar='', default=DEFAULT_DEVICE,
        help='Device for training the model. Default: gpu if available, otherwise cpu.'
    )
    parser.add_argument(
        '--dtype',
        type=get_device, metavar='', default=DEFAULT_DTYPE,
        help='Data type of the model.'
    )

    # regularizations
    parser.add_argument(
        '--l1-lambda',
        type=float, default=1e-8, metavar='',
        help='Penalty for L1 regularization. Set to 0 to disable.'
    )
    parser.add_argument(
        '--l2-lambda',
        type=float, default=0, metavar='',
        help='Penalty for L2 regularization. Set to 0 to disable.'
    )
    
    # Loading existing models
    # Used for test or when --load-to-train is True
    parser.add_argument(
        '--load-weight-path',
        type=str, default=None, metavar='',
        help='Path of the trained model to load.'
    )
    parser.add_argument(
        '--load-epoch',
        type=int, default=1, metavar='',
        help='Epoch number of the trained model to load.'
    )
    
    # training
    if not test:
        parser.add_argument(
            '--load-to-train',
            default=False, action='store_true',
            help='Whether to load existing (trained) weights for training. '
            'If False, --load-weight-path and --load-epoch are ignored.'
        )
        parser.add_argument(
            '--patience', '-p',
            type=get_patience, default=-1, metavar='',
            help='Patience for early stopping. Use -1 for no early stopping.'
        )
        
    # saving
    parser.add_argument(
        '--save-path', '-s',
        type=str, metavar='',
        help='Path to save results (weights, plots, etc.).'
    )
    return parser
    

def parse_model_settings(
    parser: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    # general
    parser.add_argument(
        '--img-height', '-H', 
        type=int, metavar='',
        help='Height of the jet images.'
    )
    parser.add_argument(
        '--img-width', '-W', 
        type=int, metavar='',
        help='Width of the jet images.'
    )
    parser.add_argument(
        '--latent-vector-size', '-L',
        type=int, metavar='',
        help='Size of the vector in the latent space.'
    )
    parser.add_argument(
        '--arxiv-model', '-a', action='store_true', default=False,
        help="Whether to use the model from the arXiv paper [arXiv:1808.08992]."
    )
    
    # encoder
    parser.add_argument(
        '--encoder-cnn-channels', '-ecc',
        nargs="+", type=int, metavar='',
        help='Channels in the encoder CNN model.'
    )
    parser.add_argument(
        '--encoder-cnn-kernel-sizes', '-eck',
        nargs="+", type=int, metavar='',
        help='Kernel sizes in the encoder CNN model.'
    )
    parser.add_argument(
        '--encoder-cnn-strides', '-ecs',
        nargs="+", type=int, metavar='', default=[1],
        help='Strides in the encoder CNN model.'
    )
    parser.add_argument(
        '--encoder-cnn-paddings', '-ecp',
        nargs="+", type=int, metavar='', default=[0],
        help='Paddings in the encoder CNN model.'
    )
    parser.add_argument(
        '--encoder-cnn-dilations', '-ecd',
        nargs="+", type=int, metavar='', default=[1],
        help='Dilations in the encoder CNN model.'
    )
    parser.add_argument(
        '--encoder-cnn-groups', '-ecg',
        nargs="+", type=int, metavar='', default=[1],
        help='Groups in the encoder CNN model.'
    )
    parser.add_argument(
        '--encoder-cnn-biases', '-ecb',
        nargs="+", type=bool, metavar='', default=[True],
        help='Biases in the encoder CNN model.'
    )
    parser.add_argument(
        '--encoder-cnn-padding-modes', '-ecpm',
        nargs="+", type=str, metavar='', default=['zeros'],
        help='Padding modes in the encoder CNN model.'
    )
    parser.add_argument(
        '--encoder-cnn-leaky-relu-negative-slopes', 
        nargs="+", type=float, metavar='', default=[0.01],
        help='Negative slopes in the leaky relu layers '
        'of the DCNN network in the encoder.'
    )
    parser.add_argument(
        '--encoder-cnn-use-intermediates',
        default=False, action='store_true',
        help='Whether to use intermediate feature maps in the encoder CNN model. '
        'If True, CNN intermediate feature maps are concatenated to the output before reducing dimension. '
    )
    
    parser.add_argument(
        '--encoder-flatten-leaky-relu-negative-slope', 
        type=float, metavar='', default=0.01,
        help='Negative slope in the leaky relu layers '
        'of the flatten layers in the encoder.'
    )
    parser.add_argument(
        '--encoder-flatten-hidden-widths',
        nargs="+", type=int, metavar='', default=[],
        help='Width of the hidden layers '
        'of the flatten layers in the encoder. '
        'If empty or None, no hidden layers are added.'
    )

    # decoder
    parser.add_argument(
        '--decoder-unflatten-channels', '-duc',
        nargs="+", type=int, metavar='',
        help='Channels in the decoder CNN model.'
    )
    parser.add_argument(
        '--decoder-cnn-channels', '-dcc',
        nargs="+", type=int, metavar='',
        help='Channels in the decoder CNN model.'
    )
    parser.add_argument(
        '--decoder-cnn-kernel-sizes', '-dck',
        nargs="+", type=int, metavar='',
        help='Kernel sizes in the decoder CNN model.'
    )
    parser.add_argument(
        '--decoder-unflatten-leaky-relu-negative-slope', 
        type=int, metavar='', default=0.01,
        help='negative_slope of the LeakyRelu layer '
        'in the unflatten layer of the decoder.'
    )
    parser.add_argument(
        '--decoder-unflatten-hidden-widths',
        nargs="+", type=int, metavar='', default=[],
        help='Width of hidden linear layers '
        'in the unflatten layer of the decoder.'
    )
    parser.add_argument(
        '--decoder-cnn-strides', '-dcs',
        nargs="+", type=int, metavar='', default=[1],
        help='Strides in the decoder CNN model.'
    )
    parser.add_argument(
        '--decoder-cnn-paddings', '-dcp',
        nargs="+", type=int, metavar='', default=[0],
        help='Paddings in the decoder CNN model.'
    )
    parser.add_argument(
        '--decoder-cnn-dilations', '-dcd',
        nargs="+", type=int, metavar='', default=[1],
        help='Dilations in the decoder CNN model.'
    )
    parser.add_argument(
        '--decoder-cnn-groups', '-dcg',
        nargs="+", type=int, metavar='', default=[1],
        help='Groups in the decoder CNN model.'
    )
    parser.add_argument(
        '--decoder-cnn-biases', '-dcb',
        nargs="+", type=bool, metavar='', default=[True],
        help='Biases in the decoder CNN model.'
    )
    parser.add_argument(
        '--decoder-cnn-padding-modes', '-dcpm',
        nargs="+", type=str, metavar='', default=['zeros'],
        help='Padding modes in the decoder CNN model.'
    )
    parser.add_argument(
        '--decoder-cnn-leaky-relu-negative-slopes',
        nargs="+", type=float, metavar='', default=[0.01],
        help='Negative slopes in the leaky relu layers '
        'of the DCNN network in the decoder.'
    )
    parser.add_argument(
        '--decoder-output-leaky-relu-negative-slope', 
        type=float, metavar='', default=0.01,
        help='Negative slope in the output leaky relu layers '
        'of the flatten layers in the encoder.'
    )
    
    return parser


def parse_plot_settings(
    parser: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    parser.add_argument(
        '--plot-start-epoch', 
        type=int, metavar='', default=100,
        help="Epoch after which the plots start."
    )
    parser.add_argument(
        '--plot-freq',
        type=int, metavar='', default=100,
        help="Frequency to plot (plot / epoch)."
    )
    
    # jet images settings
    parser.add_argument(
        '--num-jet-imgs',
        type=int, metavar='', default=10,
        help="Number of example jet images to show."
    )
    parser.add_argument(
        '--maxR',
        type=float, metavar='', default=0.6,
        help="Maximum DeltaR of the jets."
    )
    parser.add_argument(
        '--plot-cutoff',
        type=float, metavar='', default=None,
        help="The value below which the entries will be ignored. "
        "Disabled if None or non-positive, defaults to None."
    )
    return parser

###################### helper functions ######################
def get_device(arg: argparse.Namespace) -> torch.dtype:
    """Parse torch.device from input string"""
    if arg.lower() == 'cpu':
        return torch.device('cpu')
    elif 'cuda' in arg.lower() or 'gpu' in arg.lower():
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        logging.warning(
            f"{arg} is not a recognizable device. "
            f"Defaulting to {DEFAULT_DEVICE}.")
        return DEFAULT_DEVICE


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
    
def get_patience(arg: argparse.Namespace) -> int:
    """
    Parse patience from input string.
    If input is inf or smaller than 0, return inf.
    """
    if arg.lower() == 'inf':
        return inf
    else:
        p = int(arg)
        return p if p > 0 else inf