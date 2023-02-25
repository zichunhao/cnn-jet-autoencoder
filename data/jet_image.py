from typing import Optional, Union
import numpy as np
import torch
import awkward as ak
from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)

IMG_VMAX = 0.05


def pixelate(
    jet: np.ndarray,
    mask: Optional[np.ndarray] = None,
    npix: int = 64,
    maxR: float = 1.0,
) -> np.ndarray:
    """Pixelate the jet with Raghav Kansal's method.
    Reference: https://github.com/rkansal47/mnist_graph_gan/blob/neurips21/jets/final_plots.py#L191-L204

    :param jet: momenta in polar coordinates with shape (num_jets, 3, num_jet_particles)
    :type jet: np.ndarray
    :param mask: mask of data., defaults to None
    :type mask: Optional[np.ndarray], optional
    :param npix: number of pixels of the jet image., defaults to 64
    :type npix: int, optional
    :param maxR: DeltaR of the jet, defaults to 0.5
    :type maxR: float, optional
    :return: Pixelated jet with shape (npix, npix).
    :rtype: np.ndarray
    """
    bins = np.linspace(-maxR, maxR, npix + 1)
    pt = jet[:, 0]
    binned_eta = np.digitize(jet[:, 1], bins) - 1
    binned_phi = np.digitize(jet[:, 2], bins) - 1
    if mask is not None:
        pt *= mask

    jet_image = np.zeros((npix, npix))

    for eta, phi, pt in zip(binned_eta, binned_phi, pt):
        if eta >= 0 and eta < npix and phi >= 0 and phi < npix:
            jet_image[phi, eta] += pt

    return jet_image


def get_jet_rel(jets: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Get jet momenta in relative coordinates (ptrel, etarel, phirel).

    :param jets: jet momenta in polar coordinates with shape (num_jets, 3, num_jet_particles)
    :type jets: np.ndarray
    :return: jet momenta in relative coordinates with shape (num_jets, 3, num_jet_particles)
    :rtype: np.ndarray
    """
    import awkward as ak
    from coffea.nanoevents.methods import vector

    ak.behavior.update(vector.behavior)

    if isinstance(jets, torch.Tensor):
        jets = jets.detach().cpu().numpy()

    part_vecs = ak.zip(
        {
            "pt": jets[:, :, 0:1],
            "eta": jets[:, :, 1:2],
            "phi": jets[:, :, 2:3],
            "mass": np.zeros_like(jets[:, :, 1:2]),
        },
        with_name="PtEtaPhiMLorentzVector",
    )

    # sum over all the particles in each jet to get the jet 4-vector
    try:
        jet_vecs = part_vecs.sum(axis=1)[:, :2]
    except AttributeError:
        jet_vecs = ak.sum(part_vecs, axis=1)[:, :2]

    jets = normalize(jets, jet_vecs)
    return jets


def get_n_jet_images(
    jets: Union[np.ndarray, torch.Tensor],
    num_jets: int = 15,
    maxR: float = 0.5,
    npix: int = 24,
    abs_coord: int = True,
) -> np.ndarray:
    """Get the first num_jets jet images from a collection of jets.

    :param jets: Collection of jets in polar coordinates.
    :type jets: Union[np.ndarray, torch.Tensor]
    :param num_jets: umber of jet images to produce., defaults to 15
    :type num_jets: int, optional
    :param maxR: DeltaR of the jet, defaults to 0.5
    :type maxR: float, optional
    :param npix: number of pixels, defaults to 24
    :type npix: int, optional
    :param abs_coord: whether to use absolute coordinates, defaults to True.
        If False, jets are in relative coordinates.
    :type abs_coord: int, optional
    :return: The first `num_jets` jet images with shape (num_jets, npix, npix).
    :rtype: np.ndarray
    """
    if abs_coord:
        jets = get_jet_rel(jets)
    jet_image = [
        pixelate(jets[i], mask=None, npix=npix, maxR=maxR)
        for i in range(min(num_jets, len(jets)))
    ]
    jet_image = np.stack(jet_image, axis=0)
    return jet_image


def normalize(jet: np.ndarray, jet_vecs: ak.Array) -> np.ndarray:
    """Normalize jet based on jet_vecs.

    :param jet: particle features to normalize.
    :type jet: np.ndarray
    :param jet_vecs: jet feature
    :type jet_vecs: ak.Array
    :return: particle features in relative polar coordinates (pt_rel, eta_rel, phi_rel).
    :rtype: np.ndarray
    """
    # pt
    jet[:, :, 0] /= ak.to_numpy(jet_vecs.pt)
    # eta
    jet[:, :, 1] -= ak.to_numpy(jet_vecs.eta)
    # phi
    jet[:, :, 2] -= ak.to_numpy(jet_vecs.phi)
    # modulus so that phi is in [-pi, pi)
    jet[:, :, 2] = (jet[:, :, 2] + np.pi) % (2 * np.pi) - np.pi
    return jet
