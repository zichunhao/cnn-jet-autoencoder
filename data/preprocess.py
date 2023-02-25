import argparse
import logging
import jetnet
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from sklearn.model_selection import train_test_split

from jet_image import get_n_jet_images
from utils import get_dtype, DEFAULT_DTYPE


def prepare(
    jet_type: str,
    save_dir: Union[str, Path],
    test_portion: float = 0.2,
    maxR: float = 0.5,
    npix: int = 40,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare data for training and testing.

    :param jet_type: String description of the jet type,
    such as 'g' for gluon jets and 't' for top quark jets.
    :type jet_type: str
    :param save_dir: Directory to save the data.
    :type save_dir: Union[str, Path]
    :param test_portion: Portion of the data to be used for testing, defaults to 0.2
    :type test_portion: float, optional
    :param maxR: Max deltaR of the jet, defaults to 0.5
    :type maxR: float, optional
    :param npix: Number of pixels, defaults to 40
    :type npix: int, optional
    :param dtype: Data type of the jet image, defaults to torch.float
    :type dtype: torch.device, optional
    :raises TypeError: If save_dir is not a string or Path object.
    :return: (img_train, img_test, img_all)
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    logging.info(f"Downloading data ({jet_type=}) from JetNet.")
    data = jetnet.datasets.JetNet(jet_type=jet_type, data_dir=save_dir / "hdf5")
    logging.info(f"Preparing data ({jet_type=}).")

    if isinstance(save_dir, Path):
        pass
    elif isinstance(save_dir, str):
        save_dir = Path(save_dir)
    else:
        raise TypeError(
            "save_path must be of type a str or pathlib.Path. "
            f"Got: {type(save_dir)}."
        )
    save_dir.mkdir(parents=True, exist_ok=True)

    jet = data.jet_data  # (type, pt, eta, mass, num_particles)
    p = data.particle_data  # (etarel, phirel, ptrel, mask)

    # jet momenta components
    Pt, Eta = jet[..., 1], jet[..., 2]
    Phi = (np.random.random(Eta.shape) * 2 * np.pi) - np.pi  # [-pi, pi]

    # particle momenta components (relative coordinates)
    eta_rel, phi_rel, pt_rel = p[..., 0], p[..., 1], p[..., 2]

    # particle momenta components (polar coordinates)
    pt = pt_rel * Pt.reshape(-1, 1)
    eta = eta_rel + Eta.reshape(-1, 1)
    phi = phi_rel + Phi.reshape(-1, 1)
    phi = (phi % (2 * np.pi)) - np.pi  # [-pi, pi]

    p_polar = np.stack([pt, eta, phi], axis=-1)

    img = get_n_jet_images(p_polar, num_jets=len(p_polar), maxR=maxR, npix=npix)

    # training-test split
    img_train, img_test = train_test_split(img, test_size=test_portion)
    img = torch.from_numpy(img).to(dtype=dtype)
    img_train = torch.from_numpy(img_train).to(dtype=dtype)
    img_test = torch.from_numpy(img_test).to(dtype=dtype)

    torch.save(img_train, save_dir / f"{jet_type}_jets_30p_img.pt")
    torch.save(img_test, save_dir / f"{jet_type}_jets_30p_img_test.pt")
    torch.save(img, save_dir / f"{jet_type}_jets_30p_img_all.pt")

    logging.info(f"Data prepared and saved to {save_dir}.")

    return img_train, img_test, img


if __name__ == "__main__":
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # args
    parser = argparse.ArgumentParser(description="Prepare dataset for CNN Autoencoder")
    parser.add_argument(
        "-j",
        "--jet_types",
        nargs="+",
        type=str,
        default=["g", "q", "t", "w", "z"],
        help="List of jet types to download and preprocess.",
    )
    parser.add_argument(
        "-r",
        "--maxRs",
        nargs="+",
        type=float,
        default=[0.6],
        help="List of deltaR of jets. "
        "Only one number implies homogeneity for all jet types.",
    )
    parser.add_argument(
        "-p", "--npix", type=int, default=40, help="Number of pixels in the images."
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save preprocessed data.",
    )
    parser.add_argument(
        "-t",
        "--test-portion",
        type=float,
        default=0.2,
        help="Test portion of the data.",
    )
    parser.add_argument(
        "--dtype",
        type=get_dtype,
        metavar="",
        default=DEFAULT_DTYPE,
        help="Test portion of the data.",
    )

    args = parser.parse_args()
    if len(args.maxRs) == 1:
        args.maxRs = args.maxRs * len(args.jet_types)
    if len(args.maxRs) != len(args.jet_types):
        raise ValueError("Number of maxRs must be equal to the number of jet types.")
    logging.info(f"{args=}")

    for jet_type, maxR in zip(args.jet_types, args.maxRs):
        prepare(
            jet_type=jet_type,
            save_dir=Path(args.save_dir),
            test_portion=args.test_portion,
            maxR=maxR,
            npix=args.npix,
            dtype=args.dtype,
        )
