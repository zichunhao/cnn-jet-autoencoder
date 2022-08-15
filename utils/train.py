import logging
import sys
import time
sys.path.insert(1, '../')
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from typing import Tuple
from argparse import Namespace
from os import path as osp
from tqdm import tqdm
import numpy as np

from .const import MODES, LOSS_BLOW_UP_THRESHOLD
from .misc import mkdir
from .plot import plot_jet_imgs
from models import CNNJetImgEncoder, CNNJetImgDecoder

def train_loop(
    args: Namespace, 
    train_loader: DataLoader, 
    valid_loader: DataLoader,
    encoder: CNNJetImgEncoder, 
    decoder: CNNJetImgDecoder,
    optimizer_encoder: Optimizer, 
    optimizer_decoder: Optimizer
) -> int:
    """Train the autoencoder.

    :type args: Namespace
    :param train_loader: Dataloader for training data.
    :type train_loader: torch.utils.data.DataLoader
    :param valid_loader: DataLoader for validation data
    :type valid_loader: torch.utils.data.DataLoader.
    :param encoder: Encoder model.
    :type encoder: CNNJetImgEncoder
    :param decoder: Decoder model.
    :type decoder: CNNJetImgDecoder
    :param optimizer_encoder: Optimizer for encoder.
    :type optimizer_encoder: torch.optim.Optimizer
    :param optimizer_decoder: Optimizer for decoder.
    :type optimizer_decoder: torch.optim.Optimizer
    :return: Best epoch number.
    :rtype: int
    """
    best_epoch = args.load_epoch if args.load_to_train else 1
    best_loss = math.inf
    num_stale_epochs = 0
    path = mkdir(args.save_path)  # make dir and return path
    start_epoch = 1 if not args.load_to_train else args.load_epoch
    
    losses_train = []
    losses_valid = []
    dts = []  # training time
    
    # paths to save the results
    paths_jet_plots = {
        'train': mkdir(osp.join(path, 'jet_images/train')),
        'valid': mkdir(osp.join(path, 'jet_images/valid'))
    }
    path_eval = mkdir(osp.join(path, 'eval'))  # for losses and dt
    path_results = {
        'train': mkdir(osp.join(path, 'results/train')),
        'valid': mkdir(osp.join(path, 'results/valid'))
    } # for reconstructed images
    
    criterion = nn.MSELoss()
    
    for ep in range(args.num_epochs):
        curr_ep = start_epoch + ep  # current epoch
        
        # training
        start = time.time()
        imgs_target_train, imgs_recons_train, latent_spaces_train, loss_train = train(
            args, train_loader, encoder, decoder, 
            optimizer_encoder, optimizer_decoder, 
            criterion, curr_ep, args.save_path
        )
        
        # validation
        imgs_target_valid, imgs_recons_valid, latent_spaces_valid, loss_valid = validate(
            args, valid_loader, encoder, decoder, 
            criterion, curr_ep, args.save_path
        )
        
        dt = time.time() - start
        
        # append results to lists
        dts.append(dt)
        losses_train.append(loss_train)
        losses_valid.append(loss_valid)
        
        # update best epoch
        if (abs(loss_valid) < best_loss):
            best_loss = loss_valid
            num_stale_epochs = 0
            best_epoch = ep + 1
            torch.save(
                encoder.state_dict(), 
                osp.join(path, "weights_encoder/best_weights_encoder.pt")
            )
            torch.save(
                decoder.state_dict(), 
                osp.join(path, "weights_decoder/best_weights_decoder.pt")
            )
        else:
            num_stale_epochs += 1
        
        # plot
        if args.plot_freq > 0:
            if (curr_ep >= args.plot_start_epoch):
                # start plotting after args.plot_start_epoch
                plot_epoch = ((ep + 1) % args.plot_freq == 0) or (num_stale_epochs == 0)
            else:
                plot_epoch = False
        
        if plot_epoch:
            for imgs_target, imgs_recons, mode in zip(
                (imgs_target_train, imgs_target_valid),
                (imgs_recons_train, imgs_recons_valid),
                ('train', 'valid')
            ):
                plot_jet_imgs(
                    imgs_target, imgs_recons,
                    num_imgs=args.num_jet_imgs, 
                    maxR=args.maxR, 
                    save_path=paths_jet_plots[mode],
                    epoch=curr_ep,
                )
            
            # save results for monitoring along the way
            torch.save(imgs_target_train, osp.join(path_results["train"], f"imgs_target_epoch_{curr_ep}.pt"))
            torch.save(imgs_recons_train, osp.join(path_results["train"], f"imgs_recons_epoch_{curr_ep}.pt"))
            torch.save(latent_spaces_train, osp.join(path_results["train"], f"latent_epoch_{curr_ep}.pt"))
            
            torch.save(imgs_target_valid, osp.join(path_results["valid"], f"imgs_target_epoch_{curr_ep}.pt"))
            torch.save(imgs_recons_valid, osp.join(path_results["valid"], f"latent_epoch_{curr_ep}.pt"))
            torch.save(latent_spaces_valid, osp.join(path_results["valid"], f"latent_epoch_{curr_ep}.pt"))
            
            np.savetxt(osp.join(path_eval, "losses_train.txt"), losses_train)
            np.savetxt(osp.join(path_eval, "losses_valid.txt"), losses_valid)
            np.savetxt(osp.join(path_eval, "dts.txt"), dts)
            
        # loggings
        total_epochs = start_epoch + args.num_epochs
        logging.info(
            f'epoch={curr_ep}/{total_epochs}, '
            f'train_loss={loss_train}, valid_loss={loss_valid}, '
            f'{dt=}s, {num_stale_epochs=}, {best_epoch=}'
        )
        
        # patience
        if (args.patience > 0) and (num_stale_epochs > args.patience):
            logging.info(
                f'Number of stale epochs reached the set patience ({args.patience}). '
                'Training breaks.'
            )
            return best_epoch
        
    return best_epoch

def train(
    args: Namespace, 
    loader: DataLoader, 
    encoder: CNNJetImgEncoder, 
    decoder: CNNJetImgDecoder, 
    optimizer_encoder: Optimizer, 
    optimizer_decoder: Optimizer,
    criterion: nn.Module,
    epoch: int, 
    save_path: str
):
    """Function to train or test the autoencoder `(encoder, decoder)`
    using the data in `loader` and optimizers
    `(optimizer_encoder, optimizer_decoder)`.

    :param encoder: Encoder.
    :type encoder: CNNJetImgEncoder
    :param decoder: Decoder.
    :type decoder: CNNJetImgDecoder
    :param optimizer_encoder: Optimizer for `encoder`.
    :type optimizer_encoder: Optimizer
    :param optimizer_decoder: Optimizer for `decoder`.
    :type optimizer_decoder: Optimizer
    :param epoch: Current epoch, used for logging and plotting.
    :type epoch: int
    :param save_path: Path to save the results.
    :type save_path: str
    :return: (imgs_target, img_recons, latent_spaces, epoch_avg_loss)
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
    """
    return _train_valid_test(
        args, loader=loader, encoder=encoder, decoder=decoder, mode='train',
        optimizer_encoder=optimizer_encoder, optimizer_decoder=optimizer_decoder, 
        criterion=criterion, epoch=epoch, save_path=save_path
    )

@torch.no_grad()
def validate(
    args: Namespace, 
    loader: DataLoader, 
    encoder: CNNJetImgEncoder, 
    decoder: CNNJetImgDecoder, 
    criterion: nn.Module,
    epoch: int, 
    save_path: str
):
    """Validate the autoencoder `(encoder, decoder)` 
    using the data in `loader`

    :param encoder: Encoder.
    :type encoder: CNNJetImgEncoder
    :param decoder: Decoder.
    :type decoder: CNNJetImgDecoder
    :param epoch: Current epoch, used for logging and plotting.
    :type epoch: int
    :param save_path: Path to save the results.
    :type save_path: str
    :return: (imgs_target, img_recons, latent_spaces, epoch_avg_loss)
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
    """
    return _train_valid_test(
        args, loader=loader, encoder=encoder, decoder=decoder, mode='valid',
        optimizer_encoder=None, optimizer_decoder=None, criterion=criterion,
        epoch=epoch, save_path=save_path
    )
    
@torch.no_grad()
def test(
    args: Namespace, 
    loader: DataLoader, 
    encoder: CNNJetImgEncoder, 
    decoder: CNNJetImgDecoder, 
    epoch: int, 
    save_path: str, 
    criterion: nn.Module
):
    """Function for test/inference.

    :param encoder: Encoder.
    :type encoder: CNNJetImgEncoder
    :param decoder: Decoder.
    :type decoder: CNNJetImgDecoder
    :param epoch: Current epoch, used for logging and plotting.
    :type epoch: int
    :param save_path: Path to save the results.
    :type save_path: str
    :return: (imgs_target, img_recons, latent_spaces, epoch_avg_loss)
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
    """    
    return _train_valid_test(
        args, loader=loader, encoder=encoder, decoder=decoder, mode='test',
        optimizer_encoder=None, optimizer_decoder=None, criterion=criterion,
        epoch=epoch, save_path=save_path
    )
    
__ALL__ = [train_loop, train, validate, test]
    
def _train_valid_test(
    args: Namespace,
    loader: DataLoader,
    encoder: CNNJetImgEncoder,
    decoder: CNNJetImgDecoder,
    mode: str,
    optimizer_encoder: Optimizer,
    optimizer_decoder: Optimizer,
    criterion: nn.Module,
    epoch: int,
    save_path: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Function to train or test the autoencoder `(encoder, decoder)`
    using the data in `loader` and optimizers
    `(optimizer_encoder, optimizer_decoder)`.
    There are three modes: 'train', 'valid' and 'test'.

    :param loader: Dataloader for training/testing data.
    :type loader: DataLoader
    :param encoder: Encoder.
    :type encoder: CNNJetImgEncoder
    :param decoder: Decoder.
    :type decoder: CNNJetImgDecoder
    :param mode: String that specifies the mode.
        - train: for training (back prop enabled)
        - valid&test: for validation (back prop disabled)
        - test: for testing (back prop disabled)
    :type mode: str
    :param optimizer_encoder: Optimizer for `encoder`.
    :type optimizer_encoder: Optimizer
    :param optimizer_decoder: Optimizer for `decoder`.
    :type optimizer_decoder: Optimizer
    :param epoch: Current epoch, used for logging and plotting.
    :type epoch: int
    :param save_path: Path to save the results.
    :type save_path: str
    :return: (imgs_target, imgs_recons, latent_spaces, epoch_avg_loss)
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
    :raises ValueError: If mode is not 'train', 'valid' or 'test'.
    """
    mode = mode.lower()
    if mode not in MODES:
        raise ValueError(f'Invalid mode: {mode}. Supported modes: {MODES}')

    if mode == 'train':
        if (optimizer_encoder is None) or (optimizer_decoder is None):
            raise ValueError("Please specify the optimizers.")
        encoder.train()
        decoder.train()
        encoder_weight_path = mkdir(osp.join(save_path, "weights_encoder"))
        decoder_weight_path = mkdir(osp.join(save_path, "weights_decoder"))
    else:
        encoder.eval()
        decoder.eval()

    # initializations for return and saving
    imgs_target = []
    imgs_recons = []
    latent_spaces = []
    epoch_total_loss = 0

    for batch in tqdm(loader):
        # get data and normalization factor
        img_target, norm_factor = batch

        # forward pass
        latent_vec = encoder(img_target)
        img_recons = decoder(latent_vec)

        # loss
        batch_loss = criterion(img_recons, img_target)
        # regularization
        if args.l1_lambda > 0:
            batch_loss = batch_loss + args.l1_lambda * (encoder.l1_norm() + decoder.l1_norm())
        if args.l2_lambda > 0:
            batch_loss = batch_loss + args.l2_lambda * (encoder.l2_norm() + decoder.l2_norm())
        
        epoch_total_loss += batch_loss.item()
        
        if batch_loss > LOSS_BLOW_UP_THRESHOLD:  # loss blows up
            # save input and output
            torch.save(img_target.detach().cpu(), osp.join(error_path, 'img_target.pt'))
            torch.save(img_recons.detach().cpu(), osp.join(error_path, 'img_recons.pt'))
            torch.save(latent_spaces.detach().cpu(), osp.join(error_path, 'latent.pt'))
            if args.normalize:
                torch.save(norm_factor, osp.join(error_path, 'norm_factor.pt'))
            # save weights
            torch.save(encoder.state_dict(), osp.join(error_path, 'encoder_weights.pt'))
            torch.save(decoder.state_dict(), osp.join(error_path, 'decoder_weights.pt')) 
            logging.error('Loss blows up. Training breaks. Program terminates.')
            
            # exit the program
            exit(1)

        # back prop for training
        if mode == 'train':
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            try:
                batch_loss.backward()
            except RuntimeError as e:
                # grad blows up
                error_path = mkdir(osp.join(save_path, 'errors'))
                # save input and output
                torch.save(img_target.detach().cpu(), osp.join(error_path, 'img_target.pt'))
                torch.save(img_recons.detach().cpu(), osp.join(error_path, 'img_recons.pt'))
                torch.save(latent_spaces.detach().cpu(), osp.join(error_path, 'latent.pt'))
                if args.normalize:
                    torch.save(norm_factor, osp.join(error_path, 'norm_factor.pt'))
                # save weights
                torch.save(encoder.state_dict(), osp.join(error_path, 'encoder_weights.pt'))
                torch.save(decoder.state_dict(), osp.join(error_path, 'decoder_weights.pt')) 
                # raise error again to stop training
                raise e
            optimizer_encoder.step()
            optimizer_decoder.step()

        # append results
        img_target = (img_target * norm_factor).detach().cpu()
        img_recons = (img_recons * norm_factor).detach().cpu()
        imgs_target.append(img_target)
        imgs_recons.append(img_recons)
        latent_spaces.append(latent_vec.detach().cpu())

    # loop ends
    # prepare for return
    imgs_target = torch.cat(imgs_target, dim=0)
    imgs_recons = torch.cat(imgs_recons, dim=0)
    latent_spaces = torch.cat(latent_spaces, dim=0)
    epoch_avg_loss = epoch_total_loss / len(loader)

    # save weights
    if mode == 'train':
        torch.save(encoder.state_dict(), osp.join(encoder_weight_path, f"epoch_{epoch}_encoder_weights.pt"))
        torch.save(decoder.state_dict(), osp.join(decoder_weight_path, f"epoch_{epoch}_decoder_weights.pt"))
    elif mode == 'test':
        test_dir = mkdir(osp.join(save_path, 'test'))
        torch.save(imgs_target, osp.join(test_dir, 'imgs_target.pt'))
        torch.save(imgs_recons, osp.join(test_dir, 'imgs_recons.pt'))
        torch.save(latent_spaces, osp.join(test_dir, 'latent_spaces.pt'))

    return imgs_target, imgs_recons, latent_spaces, epoch_avg_loss
