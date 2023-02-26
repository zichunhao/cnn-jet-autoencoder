from copy import copy
import logging
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import torch
import numpy as np
from typing import List, Optional, Union

IMG_VMAX = 1
IMG_VMIN = 1e-8


def plot_jet_imgs(
    imgs_target: Union[torch.Tensor, List[torch.Tensor]],
    imgs_recons: Union[torch.Tensor, List[torch.Tensor]],
    num_imgs: int = 10,
    maxR: float = 0.6,
    save_path: Optional[str] = None,
    epoch: Optional[int] = None,
    file_label: Optional[str] = None,
    show: bool = False,
    cutoff: bool = True,
) -> None:
    """Plot a comparison of the target and reconstructed images.
    There will be 2 images per row, (target, reconstructed).
    There are `num_imgs+1` rows, where the first row is for the average jet image.

    :param imgs_target: Target (batched) jet images,
        with shape (num_imgs, img_height, img_width).
    :type imgs_target: Union[torch.Tensor, List[torch.Tensor]]
    :param imgs_recons: Reconstructed (batched) jet images,
        with shape (num_imgs, img_height, img_width).
    :type imgs_recons: Union[torch.Tensor, List[torch.Tensor]]
    :param num_imgs: Number of individual jet images to plot, defaults to 10.
    :type num_imgs: int, optional
    :param maxR: Maximum :math:`\Delta R` of the jet image, defaults to 0.6.
    :type maxR: float, optional
    :param save_path: Path to save the plot, defaults to None.
    If None, the plot will not be saved.
    :type save_path: Optional[str], optional
    :param epoch: Current epoch, defaults to None.
    If None, epoch will not be noted in the figure name.
    :type epoch: Optional[int], optional
    :param file_label: Label for the figure name, defaults to None.
    :type file_label: str, optional
    :param show: Whether to show the image, defaults to False.
    :type show: bool, optional
    :param cutoff: Whether the entries of reconstructed image will be masked
    if they are below the minimum value of the target image, defaults to False.
    :type cutoff: bool, optional
    :raises RuntimeError: If `imgs_recons` and `img_target`
    do not have the same number of jet images.
    """

    if len(imgs_recons) != len(imgs_target):
        raise RuntimeError(
            "Reconstructed and target jet images must contain the same number of jet images. "
            f"Got: {len(imgs_recons)=} and {len(imgs_target)=}."
        )

    # no image -> directly return
    if len(imgs_target) == 0:
        return

    # type check
    imgs_target = _type_correction(imgs_target)
    imgs_recons = _type_correction(imgs_recons)

    # mean jet image
    avg_img_target = torch.mean(imgs_target, dim=0)
    avg_img_recons = torch.mean(imgs_recons, dim=0)

    # avoid invalid values due to numerical errors
    imgs_target = torch.clip(imgs_target, min=0, max=1)
    imgs_recons = torch.clip(imgs_recons, min=0, max=1)

    if cutoff:
        # the largest order of magnitude smaller than the minimum value of the target image

        mask = imgs_recons > IMG_VMIN
        imgs_recons = imgs_recons * mask

    # return to numpy
    imgs_target = imgs_target.cpu().numpy()
    imgs_recons = imgs_recons.cpu().numpy()
    avg_img_target = avg_img_target.cpu().numpy()
    avg_img_recons = avg_img_recons.cpu().numpy()

    # settings for plot
    num_rows = num_imgs + 1  # first row is for average jet images
    cm = copy(plt.cm.jet)
    cm.set_under(color="white")

    # plot jet images
    fig, axs = plt.subplots(num_rows, 2, figsize=(7.5, 3 * num_rows))

    fig_target = axs[0][0].imshow(
        avg_img_target,
        norm=LogNorm(vmin=IMG_VMIN, vmax=1),
        origin="lower",
        cmap=cm,
        interpolation="nearest",
        extent=[-maxR, maxR, -maxR, maxR],
    )
    axs[0][0].title.set_text("Average Target Jet")

    _ = axs[0][1].imshow(
        avg_img_recons,
        norm=LogNorm(vmin=IMG_VMIN, vmax=1),
        origin="lower",
        cmap=cm,
        interpolation="nearest",
        extent=[-maxR, maxR, -maxR, maxR],
    )
    axs[0][1].title.set_text("Average Reconstructed Jet")

    cbar_target = fig.colorbar(fig_target, ax=axs[0][0])
    cbar_recons = fig.colorbar(fig_target, ax=axs[0][1])
    for cbar in (cbar_target, cbar_recons):
        cbar.set_label(r"$p_\mathrm{T}$")

    for i, axs_row in enumerate(axs[1:]):
        img_target = imgs_target[i]
        img_recons = imgs_recons[i]

        # vmax = max(np.abs(img_target).max(), np.abs(img_recons).max())

        fig_target = axs_row[0].imshow(
            img_target,
            origin="lower",
            cmap=cm,
            interpolation="nearest",
            vmin=IMG_VMIN,
            extent=[-maxR, maxR, -maxR, maxR],
            vmax=IMG_VMAX,
        )
        axs_row[0].title.set_text("Target Jet")

        _ = axs_row[1].imshow(
            img_recons,
            origin="lower",
            cmap=cm,
            interpolation="nearest",
            vmin=IMG_VMIN,
            extent=[-maxR, maxR, -maxR, maxR],
            vmax=IMG_VMAX,
        )
        axs_row[1].title.set_text("Reconstructed Jet")
        cbar_target = fig.colorbar(fig_target, ax=axs_row[0])
        cbar_recons = fig.colorbar(fig_target, ax=axs_row[1])

        for cbar in (cbar_target, cbar_recons):
            cbar.set_label(r"$p_\mathrm{T}$")

    # save figure
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig_name = "jet_images"
        if epoch is not None:
            fig_name += f"_epoch_{epoch}"
        if (file_label is not None) and (len(file_label) > 0):
            fig_name += f"_{file_label}"
        fig_name += ".pdf"
        try:
            fig.savefig(save_path / fig_name)
        except ValueError as e:
            logging.error("Failed to save jet image figure:")
            logging.error(e)
            torch.save(imgs_target, save_path / "imgs_target.pt")
            torch.save(imgs_recons, save_path / "imgs_recons.pt")
            logging.info(f"Saved jet images as .pt files in {save_path}.")
    if show:
        plt.show()
    plt.close()

    return


def _type_correction(
    jet_images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]
) -> torch.Tensor:
    if isinstance(jet_images, torch.Tensor):
        return jet_images

    elif isinstance(jet_images, list):
        if isinstance(jet_images[0], torch.Tensor):
            # list of torch.Tensor
            return torch.cat(jet_images, dim=0)
        elif isinstance(jet_images[0], np.ndarray):
            # list of numpy array
            return torch.from_numpy(np.stack(jet_images, axis=0))
        else:
            raise TypeError(
                "jet_images must be a list of torch.Tensor. "
                f"Got: {type(jet_images)}."
            )

    elif isinstance(jet_images, np.ndarray):
        # numpy array
        return torch.from_numpy(jet_images)

    else:
        # invalid type
        raise TypeError(
            "jet_images must be either torch.Tensor or np.ndarray. "
            f"Got: {type(jet_images)}."
        )
