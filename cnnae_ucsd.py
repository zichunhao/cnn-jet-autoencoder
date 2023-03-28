import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Optional, Union
from data.jet_image import get_n_jet_images as get_jet_images

import jetnet
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    Conv2D,
    Conv2DTranspose,
    Reshape,
    Flatten,
    Softmax,
)
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn import metrics

JET_FULL_NAME = {
    "qcd": "QCD",
    "g": "gluon",
    "q": "light quark",
    "t": "top quark",
    "w": "W boson",
    "z": "Z boson",
}

torch.autograd.set_detect_anomaly(True)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def plot_jet_images(
    images: List[np.ndarray],
    titles: List[str],
    filename: Union[str, Path] = "jet_image.pdf",
) -> None:
    if len(images) != len(titles):
        raise ValueError("Length of images and titles must be the same")

    n_cols = len(images)
    n_rows = len(images[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

    for r in range(n_rows):
        for c, (image, title) in enumerate(zip(images, titles)):
            if r == 0:
                axes[0][c].set_title(title, fontsize=15)
            if image.ndim == 4:
                # remove channel dimension
                image = image.squeeze(axis=-1)
            ax = axes[r][c]
            img = ax.imshow(image[r], origin="lower", norm=LogNorm(vmin=1e-3, vmax=1))
            cbar = plt.colorbar(img, ax=ax)
            cbar.set_label(r"$p_T^{rel}$", fontsize=15)
            ax.set_xlabel(r"$\Delta\eta$ cell", fontsize=15)
            ax.set_ylabel(r"$\Delta\phi$ cell", fontsize=15)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.close()
    return

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for np types 
    Source: https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16,np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(
    proj_dir: Union[Path, str],  # project directory
    data_dir: Union[Path, str],  # data directory (for storing downloaded data)
    latent_dim: int = 16,  # latent dimension of the autoencoder
    im_size: int = 40,  # jet image size
    maxR: float = 0.8,  # delta R for jet images
    num_jets: Optional[
        Union[int, float]
    ] = None,  # number of jets to use for training (None or -1 for all)
    batch_size: int = 64,  # batch size
    total_epochs: int = 2000,  # total number of epochs
    patience: int = 50,  # patience for early stopping
    verbose: int = 2,  # verbosity level for training
    num_imgs_plot: int = 6,  # number of images to plot
):
    """Script for CNNAE on JetNet dataset,
    adapted from https://github.com/jmduarte/phys139_239/blob/d687cc72cf57661f89d0a14c20e79e00026d36a7/notebooks/07_Autoencoder.ipynb

    :param proj_dir: project directory
    :param data_dir: data directory (for storing downloaded data)
    :param latent_dim: latent dimension of the autoencoder
    :param im_size: jet image size
    :param maxR: delta R for jet images
    :param num_jets: number of jets to use for training (None or -1 for all)
    :param batch_size: batch size
    :param total_epochs: total number of epochs
    :param patience: patience for early stopping
    :param verbose: verbosity level for training
    :param num_imgs_plot: number of images to plot
    """

    proj_dir = Path(proj_dir)
    proj_dir.mkdir(exist_ok=True, parents=True)
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    # prepare data
    jet_types = ("g", "q", "t", "w", "z")
    jet_imgs = {}
    jet_imgs_norm = {}
    for jet_type in jet_types:
        logging.info(f"Preparing data for {jet_type} jets...")
        data = jetnet.datasets.JetNet(jet_type=jet_type, data_dir=data_dir)
        try:
            p = data.particle_data  # (etarel, phirel, ptrel, mask)
        except AttributeError as e:
            # upgrade to latest version of jetnet
            import pip

            pip.main(["install", "--upgrade", "jetnet"])
            data = jetnet.datasets.JetNet(jet_type=jet_type, data_dir=data_dir)
            p = data.particle_data  # (etarel, phirel, ptrel, mask)

        # particle momenta components (relative coordinates)
        eta_rel, phi_rel, pt_rel = p[..., 0], p[..., 1], p[..., 2]
        p_polar = np.stack([pt_rel, eta_rel, phi_rel], axis=-1)
        img = get_jet_images(
            jets=p_polar,
            num_jets=len(p_polar),
            maxR=maxR,
            npix=im_size,
            abs_coord=False,  # relative coordinates already
        )
        # img = jetnet.utils.to_image(p_polar, im_size=im_size, maxR=maxR)
        img = np.expand_dims(img, axis=-1)
        img_sum = np.sum(img, axis=(1, 2), keepdims=True)
        jet_imgs_norm[jet_type] = img_sum
        jet_imgs[jet_type] = np.copy(img / img_sum)

        del data, p, eta_rel, phi_rel, pt_rel, p_polar

    # combine g and q jets to make qcd jets
    jet_imgs["qcd"] = np.concatenate([jet_imgs["g"], jet_imgs["q"]], axis=0)
    jet_imgs.pop("g")
    jet_imgs.pop("q")
    jet_types = list(jet_imgs.keys())
    logging.info(f"jet types: {jet_types}")

    # training-validation-test split
    jet_imgs_train = {}
    jet_imgs_val = {}
    jet_imgs_test = {}
    for jet_type in jet_types:
        # jet_img_train, jet_imgs_test[jet_type] = train_test_split(
        #     jet_imgs[jet_type], test_size=0.2, random_state=42
        # )
        train_valid_size = int(0.8 * len(jet_imgs[jet_type]))
        jet_img_train, jet_imgs_test[jet_type] = jet_imgs[jet_type][:train_valid_size], jet_imgs[jet_type][train_valid_size:]
        # train/validation split
        if num_jets is not None and num_jets > 0:
            if num_jets < 1:
                # use fraction of jets
                num_jets = int(num_jets * len(jet_img_train))
            else:
                try:
                    num_jets = int(num_jets)
                except ValueError as e:
                    # use all jets
                    logging.error(f"Error when parsing num_jets: {e}")
                    logging.error("Using all jets instead")
                    num_jets = len(jet_img_train)
            jet_img_train = jet_img_train[:num_jets]
        
        jet_imgs_train[jet_type], jet_imgs_val[jet_type] = train_test_split(
            jet_img_train, test_size=0.25, random_state=42
        )

        del jet_img_train

    # create autoencoder
    logging.info("Creating autoencoder")
    x_in = Input(shape=(im_size, im_size, 1))
    x = Conv2D(
        128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"
    )(x_in)
    x = Conv2D(
        128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"
    )(x)
    x = Flatten()(x)

    x_enc = Dense(latent_dim, name="bottleneck")(x)

    x = Dense(int(im_size * im_size / 16) * 128, activation="relu")(x_enc)
    x = Reshape((int(im_size / 4), int(im_size / 4), 128))(x)
    x = Conv2DTranspose(
        128, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same"
    )(x)
    x = Conv2DTranspose(
        1, kernel_size=(3, 3), strides=(2, 2), activation="linear", padding="same"
    )(x)
    x_out = Softmax(name="softmax", axis=[-2, -3])(x)
    model = Model(inputs=x_in, outputs=x_out, name="autoencoder")

    model.compile(loss="mse", optimizer="adam")
    model.summary()

    # save the encoder-only model for easy access to latent space
    encoder = Model(inputs=x_in, outputs=x_enc, name="encoder")

    compression_rate = im_size * im_size / latent_dim
    logging.info(f"Compression rate: {compression_rate}%")

    # train autoencoder
    logging.info("Training autoencoder with early stopping...")
    history = model.fit(
        jet_imgs_train["qcd"],
        jet_imgs_train["qcd"],
        batch_size=batch_size,
        epochs=total_epochs,
        verbose=verbose,
        validation_data=(jet_imgs_val["qcd"], jet_imgs_val["qcd"]),
        callbacks=EarlyStopping(monitor="val_loss", patience=patience),
    )

    model.save(proj_dir / "model.h5")

    # evaluate autoencoder

    fig_dir = proj_dir / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    # reconstruction
    images_target = {}
    images_recons = {}
    for jet_type in jet_types:
        logging.info(f"Reconstructing {jet_type} images from test set...")
        target = jet_imgs_test[jet_type]
        recons = model.predict(target)
        images_target[jet_type] = target
        images_recons[jet_type] = recons

        # target_avg = np.mean(target, axis=0)
        # recons_avg = np.mean(recons, axis=0)

    torch.save(images_target, proj_dir / "images_target.pt")
    torch.save(images_recons, proj_dir / "images_recons.pt")

    for jet_type in jet_types:
        target = images_target[jet_type]
        recons = images_recons[jet_type]
        plot_jet_images(
            images=[target[:num_imgs_plot], recons[:num_imgs_plot]],
            titles=[
                f"Target {JET_FULL_NAME[jet_type]} jet image",
                f"Reconstructed {JET_FULL_NAME[jet_type]} jet image",
            ],
            filename=fig_dir / f"jet_images_{jet_type}.pdf",
        )

    # anomaly detection
    ae_stats = {}
    logging.info("Calculating anomaly scores (MSE)...")
    scores = {}
    for jet_type in jet_types:
        mse = np.power((images_recons[jet_type] - images_target[jet_type]), 2)
        mse = np.sum(mse.reshape(-1, im_size * im_size), axis=-1)
        scores[jet_type] = mse
    torch.save(scores, proj_dir / "anomaly_scores.pt")

    scores_bkg = scores["qcd"]

    for jet_type in jet_types:
        if jet_type.lower() == "qcd":
            continue
        logging.info(f"Plotting ROC curve for qcd vs. {jet_type}...")
        scores_sig = scores[jet_type]
        fpr, tpr, threshold = metrics.roc_curve(
            np.concatenate([-np.ones(len(scores_bkg)), np.ones(len(scores_sig))]),
            np.concatenate([scores_bkg, scores_sig]),
        )
        auc = metrics.auc(fpr, tpr)
        tpr_10_percent = tpr[np.searchsorted(fpr, 0.1)]
        tpr_1_percent = tpr[np.searchsorted(fpr, 0.01)]

        if auc < 0.5:
            fpr, tpr, _ = metrics.roc_curve(
                np.concatenate([np.ones(len(scores_bkg)), -np.ones(len(scores_sig))]),
                np.concatenate([scores_bkg, scores_sig]),
            )
            auc = metrics.auc(fpr, tpr)
            
        ae_stats["all"] = {
            "tpr": tpr,
            "fpr": fpr,
            "auc": auc,
            "tpr @ fpr=10%": tpr_10_percent,
            "tpr @ fpr=1%": tpr_1_percent,
        }

        plt.plot(tpr, fpr, label="AUC = {:.4f}".format(auc))
        plt.xlabel(r"$\varepsilon_s$")
        plt.ylabel(r"$\varepsilon_b$")
        plt.yscale("log")
        plt.ylim(1e-3, 1)
        plt.grid(linestyle="--", linewidth=0.5)
        plt.legend()

        plt.savefig(fig_dir / f"roc_{jet_type}.pdf")
        plt.close()

    # qcd vs all
    logging.info("Plotting ROC curve for qcd vs. all...")
    scores_sig = np.concatenate(
        [scores[jet_type] for jet_type in jet_types if jet_type.lower() != "qcd"]
    )
    fpr, tpr, threshold = metrics.roc_curve(
        np.concatenate([-np.ones(len(scores_bkg)), np.ones(len(scores_sig))]),
        np.concatenate([scores_bkg, scores_sig]),
    )
    auc = metrics.auc(fpr, tpr)

    if auc < 0.5:
        fpr, tpr, _ = metrics.roc_curve(
            np.concatenate([np.ones(len(scores_bkg)), -np.ones(len(scores_sig))]),
            np.concatenate([scores_bkg, scores_sig]),
        )
        auc = metrics.auc(fpr, tpr)
    
    tpr_10_percent = tpr[np.searchsorted(fpr, 0.1)]
    tpr_1_percent = tpr[np.searchsorted(fpr, 0.01)]

    ae_stats["all"] = {
        "tpr": tpr,
        "fpr": fpr,
        "auc": auc,
        "tpr @ fpr=10%": tpr_10_percent,
        "tpr @ fpr=1%": tpr_1_percent,
    }

    plt.plot(tpr, fpr, label="AUC = {:.4f}".format(auc))
    plt.xlabel(r"$\varepsilon_s$")
    plt.ylabel(r"$\varepsilon_b$")
    plt.yscale("log")
    plt.ylim(1e-3, 1)
    plt.xlim(0, 1)
    plt.grid(linestyle="--", linewidth=0.5)
    plt.legend()

    plt.savefig(fig_dir / f"roc_all.pdf")
    plt.close()

    json.dump(ae_stats, open(proj_dir / "anomaly_detection.json", "w"), cls=NumpyEncoder)
    torch.save(ae_stats, proj_dir / "anomaly_detection.pt")

    logging.info("Anomaly detection done")
    return


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train CNN autoencoder")

    argparser.add_argument(
        "--proj-dir", type=str, required=True, help="Project directory"
    )
    argparser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    argparser.add_argument(
        "--latent-dim", type=int, default=16, help="Latent dimension of the autoencoder"
    )
    argparser.add_argument("--im-size", type=int, default=40, help="Jet image size")
    argparser.add_argument(
        "--maxR", type=float, default=0.8, help="Delta R for jet images"
    )
    argparser.add_argument(
        "--num-jets",
        type=float,
        default=-1,
        help="Number of jets to use for training (None or -1 for all)",
    )
    argparser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    argparser.add_argument(
        "--total-epochs", type=int, default=2000, help="Total number of epochs"
    )
    argparser.add_argument(
        "--patience", type=int, default=50, help="Patience for early stopping"
    )
    argparser.add_argument(
        "--verbose", type=int, default=2, help="Verbosity level for training"
    )
    argparser.add_argument(
        "--num-imgs-plot", type=int, default=6, help="Number of images to plot"
    )

    args = argparser.parse_args()
    logging.info(f"{args=}")

    logging.info("Running script...")
    main(
        proj_dir=args.proj_dir,
        data_dir=args.data_dir,
        latent_dim=args.latent_dim,
        im_size=args.im_size,
        maxR=args.maxR,
        num_jets=args.num_jets,
        batch_size=args.batch_size,
        total_epochs=args.total_epochs,
        patience=args.patience,
        verbose=args.verbose,
        num_imgs_plot=args.num_imgs_plot,
    )
    logging.info("Done!")
