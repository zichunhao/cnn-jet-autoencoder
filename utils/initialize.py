
from models import CNNJetImgEncoder, CNNJetImgDecoder, CNNAEFNSEncoder, CNNAEFNSDecoder
from argparse import Namespace
from typing import Tuple, Union
from torch.optim import Optimizer
import torch
from torch.utils.data import DataLoader

import sys

from .dataset import JetImageDataset
from sklearn.model_selection import train_test_split
sys.path.insert(0, '../')


def initialize_autoencoder(
    args: Namespace
) -> Union[Tuple[CNNJetImgEncoder, CNNJetImgDecoder],
           Tuple[CNNAEFNSEncoder, CNNAEFNSDecoder]]:
    """Initialize the encoder and decoder."""
    if args.arxiv_model:
        # arXiv:1808.08992
        return CNNAEFNSEncoder(), CNNAEFNSDecoder()
    else:
        
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
            cnn_use_intermediates=args.encoder_cnn_use_intermediates,
            flatten_leaky_relu_negative_slope=args.encoder_flatten_leaky_relu_negative_slope,
            flatten_hidden_widths=args.encoder_flatten_hidden_widths,
            device=args.device,
            dtype=args.dtype
        )
        decoder = CNNJetImgDecoder(
            latent_vector_size=args.latent_vector_size,
            output_height=args.img_height,
            output_width=args.img_width,
            unflatten_img_size=encoder.dcnn_out_img_size[1:],
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


def initialize_optimizers(
    args: Namespace,
    encoder: CNNJetImgEncoder,
    decoder: CNNJetImgDecoder
) -> Tuple[Optimizer, Optimizer]:
    """Initialize optimizers for the encoder and decoder."""    
    if args.optimizer.lower() == 'adam':
        optimizer_encoder = torch.optim.Adam(encoder.parameters(), args.learning_rate)
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), args.learning_rate)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer_encoder = torch.optim.RMSprop(encoder.parameters(), lr=args.learning_rate)
        optimizer_decoder = torch.optim.RMSprop(decoder.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == 'sgd':
        optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=args.learning_rate)
        optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=args.learning_rate)
    # TODO: add more supported optimizers if necessary
    else:
        raise NotImplementedError(
            "Available choices are 'Adam', 'RMSprop', and 'SGD'. "
            f"Found: {args.optimizer}."
        )
    return optimizer_encoder, optimizer_decoder


def initialize_dataloader(args: Namespace, test: bool=False) -> DataLoader:
    """Initialize the dataloader for training

    :param test: Whether the data is for testing/inference purpose, defaults to False
        If True, data will not be shuffled.
    :type test: bool, optional
    :return: (train_loader, valid_loader) for train (`not test`) and test_loader if `test`
    :rtype: DataLoader
    """
    # load data
    jet_imgs = []
    for path in args.data_paths:
        jet_imgs.append(torch.load(path).to(device=args.device, dtype=args.dtype))
    jet_imgs = torch.cat(jet_imgs, dim=0)
    
    shuffle = not test  # do not shuffle if testing
    if not test:  # training
        jet_imgs_train, jet_imgs_valid = train_test_split(
            jet_imgs,
            test_size=args.test_size
        )
        dataset_train = JetImageDataset(
            jet_imgs=jet_imgs_train, 
            normalize=args.normalize, 
            shuffle=shuffle, 
            device=args.device, 
            dtype=args.dtype
        )
        loader_train = DataLoader(
            dataset=dataset_train, 
            batch_size=args.batch_size, 
            shuffle=shuffle
        )
        dataset_valid = JetImageDataset(
            jet_imgs=jet_imgs_valid, 
            normalize=args.normalize, 
            shuffle=shuffle, 
            device=args.device, 
            dtype=args.dtype
        )
        loader_valid = DataLoader(
            dataset=dataset_valid, 
            batch_size=args.batch_size, 
            shuffle=shuffle
        )
        return (loader_train, loader_valid)
    else:
        dataset = JetImageDataset(
            jet_imgs=jet_imgs, 
            normalize=args.normalize, 
            shuffle=shuffle, 
            device=args.device, 
            dtype=args.dtype
        )
        return DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle)