from pathlib import Path
from utils import (
    anomaly_detection_ROC_AUC,
    plot_jet_imgs,
    test,
    initialize_autoencoder,
    initialize_dataloader,
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

from utils import ARXIV_MODEL_LATENT_VECTOR_SIZE, get_eps


def main(args: argparse.Namespace):
    logging.info(f'{args=}')

    # initializations
    logging.info('Initialize dataloader.')
    dataloader = initialize_dataloader(args, test=True)

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
    
    path_test = Path(args.save_path) / 'test'
    path_test.mkdir(parents=True, exist_ok=True)

    # test
    imgs_target, imgs_recons, latent_spaces, norm_factors, epoch_avg_loss = test(
        args,
        loader=dataloader,
        encoder=encoder,
        decoder=decoder,
        epoch=args.load_epoch,  # not used
        save_path=path_test,
        criterion=torch.nn.MSELoss(),
    )
    torch.save(imgs_target, path_test / 'imgs_target.pt')
    torch.save(imgs_recons, path_test / 'imgs_recons.pt')
    torch.save(norm_factors, path_test / 'norm_factors.pt')
    torch.save(latent_spaces, path_test / 'latent_spaces.pt')
    logging.info(f'epoch_avg_loss: {epoch_avg_loss}')
    
    plot_jet_imgs(
        imgs_target, imgs_recons,
        num_imgs=args.num_jet_imgs, 
        maxR=args.maxR, 
        save_path=path_test,
        epoch=None,
        file_label=args.data_label,
        cutoff=args.plot_cutoff
    )
    
    # anomaly detection
    if (args.anomaly_detection) and (len(args.signal_paths) > 0):
        eps = get_eps(args)
        logging.info('Anomaly detection')
        path_ad = path_test / "anomaly_detection"
        path_ad.mkdir(parents=True, exist_ok=True)
        bkg_recons, bkg_target, bkg_norms = imgs_target, imgs_recons, norm_factors
        bkg_recons_normalized = bkg_recons / (bkg_norms + eps)
        bkg_target_normalized = bkg_target / (bkg_norms + eps)
        
        sig_recons_list = []
        sig_target_list = []
        sig_norms_list = []
        sig_recons_normalized_list = []
        sig_target_normalized_list = []
        
        for signal_path, signal_type in zip(args.signal_paths, args.signal_labels):
            # background vs single signal
            logging.info(f'Anomaly detection: {args.data_label} vs. {signal_type}')
            path_ad_single = path_ad / f"single_signals/{signal_type}"
            path_ad_single.mkdir(parents=True, exist_ok=True)
            
            sig_loader = initialize_dataloader(
                args, 
                test=True, 
                paths=[signal_path]
            )
            sig_target, sig_recons, sig_latent, sig_norms, _ = test(
                args,
                loader=sig_loader,
                encoder=encoder,
                decoder=decoder,
                epoch=args.load_epoch,  # not used
                save_path=path_ad_single,
                criterion=torch.nn.MSELoss(),
            )

            sig_recons_normalized = sig_recons / (sig_norms + eps)
            sig_target_normalized = sig_target / (sig_norms + eps)

            anomaly_detection_ROC_AUC(
                sig_recons, sig_target, sig_recons_normalized, sig_target_normalized,
                bkg_recons, bkg_target, bkg_recons_normalized, bkg_target_normalized,
                save_path=path_ad_single,
                img_label=signal_type
            )
            
            plot_jet_imgs(
                sig_target, sig_recons,
                num_imgs=args.num_jet_imgs, 
                maxR=args.maxR, 
                save_path=path_ad_single,
                epoch=None,
                file_label=signal_type,
                cutoff=args.plot_cutoff
            )

            # add to list
            sig_recons_list.append(sig_recons)
            sig_target_list.append(sig_target)
            sig_norms_list.append(sig_norms)
            sig_recons_normalized_list.append(sig_recons_normalized)
            sig_target_normalized_list.append(sig_target_normalized)

            # save results
            torch.save(sig_recons, path_ad_single / f"{signal_type}_recons.pt")
            torch.save(sig_target, path_ad_single / f"{signal_type}_target.pt")
            torch.save(sig_norms, path_ad_single / f"{signal_type}_norms.pt")
            torch.save(sig_latent, path_ad_single / f"{signal_type}_latent.pt")

        # bkg vs. all signals
        sig_recons = torch.cat(sig_recons_list, dim=0)
        sig_target = torch.cat(sig_target_list, dim=0)
        sig_norms = torch.cat(sig_norms_list, dim=0)
        sig_recons_normalized = torch.cat(sig_recons_normalized_list, dim=0)
        sig_target_normalized = torch.cat(sig_target_normalized_list, dim=0)

        logging.info(f'Anomaly detection: {args.data_label} vs. {args.signal_labels}')
        anomaly_detection_ROC_AUC(
            sig_recons, sig_target, sig_recons_normalized, sig_target_normalized,
            bkg_recons, bkg_target, bkg_recons_normalized, bkg_target_normalized,
            save_path=path_ad
        )
    
    elif (args.anomaly_detection) and (len(args.signal_paths) > 0):
        logging.error("No signal paths given for anomaly detection.")
    
    


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='CNN Autoencoder Training Options'
    )
    parser = parse_model_settings(parser)
    parser = parse_training_settings(parser, test=True)
    parser = parse_data_settings(parser)
    parser = parse_plot_settings(parser)

    args = parser.parse_args()
    if args.arxiv_model:
        args.latent_vector_size = ARXIV_MODEL_LATENT_VECTOR_SIZE
    return args


if __name__ == '__main__':
    import sys
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    main(args)
