
from argparse import Namespace
from typing import Tuple
from torch.optim import Optimizer
import torch
from torch.utils.data import DataLoader

import sys

from .dataset import JetImageDataset
sys.path.insert('../')
from models import CNNJetImgEncoder, CNNJetImgDecoder


def intialize_autoencoder(args: Namespace) -> Tuple[CNNJetImgEncoder, CNNJetImgDecoder]:
    """Intialize the encoder and decoder."""
    encoder = CNNJetImgEncoder(
        input_height=args.img_height,
        input_width=args.img_width,
        latent_vector_size=args.latent_vector_size,
        cnn_channels=args.encoder_cnn_channels,
        cnn_kernel_sizes=args.encoder_cnn_kernel_sizes,
        cnn_strides=args.encoder_cnn_strides,
        cnn_paddings=args.encoder_cnn_paddings,
        cnn_dilations=args.encoder_cnn_dilations,
        cnn_groups=args.encoder_cnn_groups,
        cnn_biases=args.encoder_cnn_biases,
        cnn_padding_modes=args.encoder_cnn_padding_modes,
        cnn_leaky_relu_negative_slopes=args.encoder_cnn_leaky_relu_negative_slopes,
        flatten_leaky_relu_negative_slope=args.encoder_flatten_leaky_relu_negative_slope,
        flatten_hidden_widths=args.encoder_flatten_hidden_widths,
        device=args.device,
        dtype=args.dtype
    )
    decoder = CNNJetImgDecoder(
        latent_vector_size=args.latent_vector_size,
        output_height=args.img_height,
        output_width=args.img_width,
        unflatten_img_size=encoder.dcnn_out_img_size,
        cnn_channels=args.decoder_cnn_channels,
        cnn_kernel_sizes=args.decoder_cnn_kernel_sizes,
        unflatten_leaky_relu_negative_slope=args.decoder_unflatten_leaky_relu_negative_slope,
        unflatten_hidden_widths=args.decoder_unflatten_hidden_widths,
        cnn_strides=args.decoder_cnn_strides,
        cnn_paddings=args.decoder_cnn_paddings,
        cnn_dilations=args.decoder_cnn_dilations,
        cnn_groups=args.decoder_cnn_groups,
        cnn_biases=args.decoder_cnn_biases,
        cnn_padding_modes=args.decoder_cnn_padding_modes,
        cnn_leaky_relu_negative_slopes=args.decoder_cnn_leaky_relu_negative_slopes,
        output_leaky_relu_negative_slope=args.decoder_output_leaky_relu_negative_slope,
        device=args.device,
        dtype=args.dtype
    )
    return encoder, decoder


def intialize_optimizers(
    args: Namespace,
    encoder: CNNJetImgEncoder,
    decoder: CNNJetImgDecoder
) -> Tuple[Optimizer, Optimizer]:
    """Initialize optimizers for the encoder and decoder."""    
    if args.optimizer.lower() == 'adam':
        optimizer_encoder = torch.optim.Adam(encoder.parameters(), args.lr)
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), args.lr)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer_encoder = torch.optim.RMSprop(encoder.parameters(), lr=args.lr)
        optimizer_decoder = torch.optim.RMSprop(decoder.parameters(), lr=args.lr)
    # TODO: add more supported optimizers if necessary
    else:
        raise NotImplementedError(
            "Available choices are 'Adam' and 'RMSprop'. "
            f"Found: {args.optimizer}."
        )
    return optimizer_encoder, optimizer_decoder


def intialize_dataloader(args: Namespace, test: bool=False) -> DataLoader:
    """Intialize the dataloader for training

    :param test: Whether the data is for testing/inference purpose, defaults to False
        If True, data will not be shuffled.
    :type test: bool, optional
    :return: _description_
    :rtype: DataLoader
    """    
    jet_imgs = torch.load(args.data_path).to(device=args.device, dtype=args.dtype)
    shuffle = not test  # do not shuffle if testing
    dataset = JetImageDataset(
        jet_imgs=jet_imgs, 
        normalize=args.normalize, 
        shuffle=shuffle, 
        device=args.device, 
        dtype=args.dtype
    )
    return DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle)