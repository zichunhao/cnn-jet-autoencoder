import argparse
import torch
from .const import DEFAULT_DEVICE, DEFAULT_DTYPE

def parse_training_settings(
    parser: argparse.ArgumentParser
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
        '--device',
        type=torch.device, metavar='', default=DEFAULT_DEVICE,
        help='Device for training the model.'
    )
    parser.add_argument(
        '--dtype',
        type=torch.dtype, metavar='', default=DEFAULT_DEVICE,
        help='Data type of the model.'
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
        '--encoder-flatten-leaky-relu-negative-slopes', 
        nargs="+", type=float, metavar='', default=[0.01],
        help='Negative slopes in the leaky relu layers '
        'of the flatten layers in the encoder.'
    )
    parser.add_argument(
        '--encoder-flatten-hidden-width',
        nargs="+", type=int, metavar='', default=[],
        help='Width of the hidden layers '
        'of the flatten layers in the encoder. '
        'If empty or None, no hidden layers are added.'
    )

    # decoder
    parser.add_argument(
        '--decoder-unflatten-channels', '-dcc',
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
        '--decoder-unflatten-hidden-width',
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
        '--decoder-output-leaky-relu-negative-slopes', 
        nargs="+", type=float, metavar='', default=[0.01],
        help='Negative slopes in the output leaky relu layers '
        'of the flatten layers in the encoder.'
    )
    
    return parser