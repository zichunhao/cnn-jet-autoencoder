from utils import (
    train_loop,
    initialize_autoencoder,
    initialize_dataloader,
    initialize_optimizers,
    parse_model_settings,
    parse_training_settings,
    parse_data_settings,
    parse_plot_settings,
    get_compression_rate
)
    

import torch
import math
import logging
import argparse

def main(args: argparse.Namespace):
    logging.info(f'{args=}')
    
    logging.info('Initialize dataloader')
    loader_train, loader_valid = initialize_dataloader(args, test=False)
    
    logging.info('Initialize models')
    encoder, decoder = initialize_autoencoder(args)
    logging.info(f'{encoder=}')
    logging.info(f'{decoder=}')
    logging.info(f'{encoder.num_learnable_parameters=}')
    logging.info(f'{decoder.num_learnable_parameters=}')
    compression_rate = get_compression_rate(
        img_height=args.img_height, 
        img_width=args.img_width, 
        latent_vector_size=args.latent_vector_size
    )
    logging.info(f'compression rate: {compression_rate}')
    
    logging.info('Initialize optimizers')
    optim_encoder, optim_decoder = initialize_optimizers(
        args, encoder, decoder
    )
    
    best_ep = train_loop(
        args, 
        train_loader=loader_train, 
        valid_loader=loader_valid,
        encoder=encoder,
        decoder=decoder,
        optimizer_encoder=optim_encoder,
        optimizer_decoder=optim_decoder,
        lambda_nz=args.lambda_nz
    )
    logging.info(f'Training completed. Best epoch: {best_ep}')


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='CNN Autoencoder Training Options'
    )
    parser = parse_model_settings(parser)
    parser = parse_training_settings(parser)
    parser = parse_data_settings(parser)
    parser = parse_plot_settings(parser)
    
    args = parser.parse_args()
    if args.patience < 0:
        args.patience = math.inf
    return args


if __name__ == '__main__':
    import sys
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    main(args)