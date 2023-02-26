from pathlib import Path
from sklearn import metrics
from typing import Dict, List, Optional, Tuple, Union
from matplotlib import pyplot as plt
import torch
import numpy as np

MSE = torch.nn.MSELoss(reduction="none")
KEY_MSE = "MSE"
KEY_MSE_NORM = "MSE (normalized jet images)"


def anomaly_detection_ROC_AUC(
    sig_recons: torch.Tensor,
    sig_target: torch.Tensor,
    sig_recons_normalized: torch.Tensor,
    sig_target_normalized: torch.Tensor,
    bkg_recons: torch.Tensor,
    bkg_target: torch.Tensor,
    bkg_recons_normalized: torch.Tensor,
    bkg_target_normalized: torch.Tensor,
    save_path: Union[str, Path] = None,
    plot_rocs: bool = True,
    rocs_hlines: List[float] = [1e-1, 1e-2],
    img_label: Optional[str] = None,
) -> Tuple[Dict[str, Tuple[np.ndarray]], Dict[str, Tuple[np.ndarray]]]:
    """Compute get AUC and ROC curves in the anomaly detection.

    :param sig_recons: Reconstructed signal jets.
    :type sig_recons: torch.Tensor
    :param sig_target: Target signal jets.
    :type sig_target: torch.Tensor
    :param sig_recons_normalized: Reconstructed normalized signal jets.
    :type sig_recons_normalized: torch.Tensor
    :param sig_target_normalized: Reconstructed normalized signal jets.
    :type sig_target_normalized: torch.Tensor
    :param bkg_recons: Reconstructed background jets.
    :type bkg_recons: torch.Tensor
    :param bkg_target: Target background jets.
    :type bkg_target: torch.Tensor
    :param bkg_recons_normalized: Reconstructed normalized background jets.
    :type bkg_recons_normalized: torch.Tensor
    :param bkg_target_normalized: Target normalized background jets.
    :type bkg_target_normalized: torch.Tensor
    :param save_path: Path to save the ROC curves and AUCs, defaults to None.
    If None, the ROC curves and AUCs are not saved.
    :type save_path: str, optional
    :param rocs_hlines: Horizontal lines (and intercept) to plot on the ROC curves,
    defaults to [1e-1, 1e-2].
    :type rocs_hlines: List[float], optional
    :param img_label: Label to add to img file, defaults to None.
    :type img_label: Optional[str], optional
    :return: (`roc_curves`, `aucs`),
    where `roc_curves` is a dictionary {kind: roc_curve},
    and `aucs` is a dictionary {kind: auc}.
    :rtype: Tuple[Dict[str, Tuple[np.ndarray]], Dict[str, Tuple[np.ndarray]]]
    """
    scores_dict, true_labels = anomaly_scores_sig_bkg(
        sig_recons,
        sig_target,
        sig_recons_normalized,
        sig_target_normalized,
        bkg_recons,
        bkg_target,
        bkg_recons_normalized,
        bkg_target_normalized,
    )
    roc_curves = dict()
    aucs = dict()
    for kind, scores in scores_dict.items():
        roc_curve = metrics.roc_curve(true_labels, scores)
        roc_curves[kind] = roc_curve
        auc = metrics.auc(roc_curve[0], roc_curve[1])
        if auc < 0.5:
            # opposite labels
            roc_curve = metrics.roc_curve(-true_labels, scores)
            roc_curves[kind] = roc_curve
            auc = metrics.auc(roc_curve[0], roc_curve[1])
        aucs[kind] = auc

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save(scores_dict, save_path / "scores.pt")
        torch.save(true_labels, save_path / "true_labels.pt")
        torch.save(roc_curves, save_path / "roc_curves.pt")
        torch.save(aucs, save_path / "aucs.pt")

    if plot_rocs:
        auc_sorted = list(sorted(aucs.items(), key=lambda x: x[1], reverse=True))

        def plot_roc_curves(auc, path: Union[str, Path] = None):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.set_xlabel("True Positive Rate")
            ax.set_ylabel("False Positive Rate")
            ax.set_yscale("log")

            for kind, auc in auc:
                fpr, tpr, thresholds = roc_curves[kind]
                ax.plot(tpr, fpr, label=f"{kind} (AUC: {auc:.5f})")
                for y_value in rocs_hlines:
                    ax.plot(
                        np.linspace(0, 1, 100),
                        [y_value] * 100,
                        "--",
                        c="gray",
                        linewidth=1,
                    )
                    ax.vlines(
                        x=tpr[np.searchsorted(fpr, y_value)],
                        ymin=0,
                        ymax=y_value,
                        linestyles="--",
                        colors="gray",
                        linewidth=1,
                    )

                fpr, tpr, thresholds = roc_curves[kind]

            plt.legend()
            if path is not None:
                plt.savefig(path)

            return fig

        if save_path is not None:
            file_name = "roc_curves"
            if img_label is not None:
                file_name += f"_{img_label}"
            file_name += ".pdf"
            plot_roc_curves(auc_sorted, path=save_path / file_name)
        else:
            plot_roc_curves(auc_sorted, path=None)

    return roc_curves, aucs


def anomaly_scores_sig_bkg(
    sig_recons: torch.Tensor,
    sig_target: torch.Tensor,
    sig_recons_normalized: torch.Tensor,
    sig_target_normalized: torch.Tensor,
    bkg_recons: torch.Tensor,
    bkg_target: torch.Tensor,
    bkg_recons_normalized: torch.Tensor,
    bkg_target_normalized: torch.Tensor,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute anomaly scores for signal and background
    and return the anomaly scores along with the true labels.

    :param sig_recons: Reconstructed signal jets.
    :type sig_recons: torch.Tensor
    :param sig_target: Target signal jets.
    :type sig_target: torch.Tensor
    :param sig_recons_normalized: Reconstructed normalized signal jets.
    :type sig_recons_normalized: torch.Tensor
    :param sig_target_normalized: Reconstructed normalized signal jets.
    :type sig_target_normalized: torch.Tensor
    :param bkg_recons: Reconstructed background jets.
    :type bkg_recons: torch.Tensor
    :param bkg_target: Target background jets.
    :type bkg_target: torch.Tensor
    :param bkg_recons_normalized: Reconstructed normalized background jets.
    :type bkg_recons_normalized: torch.Tensor
    :param bkg_target_normalized: Target normalized background jets.
    :type bkg_target_normalized: torch.Tensor
    :return: (true_labels, scores), where scores is a dictionary
    with the scores (value) for each type (key).
    :rtype: Tuple[np.ndarray, Dict[str, np.ndarray]]
    """
    sig_scores = anomaly_scores(
        sig_recons,
        sig_target,
        sig_recons_normalized,
        sig_target_normalized,
    )
    bkg_scores = anomaly_scores(
        bkg_recons,
        bkg_target,
        bkg_recons_normalized,
        bkg_target_normalized,
    )
    scores = {
        k: np.concatenate([sig_scores[k], bkg_scores[k]]) for k in sig_scores.keys()
    }
    true_labels = np.concatenate(
        [np.ones_like(sig_scores[KEY_MSE]), -np.ones_like(bkg_scores[KEY_MSE])]
    )
    return scores, true_labels


def anomaly_scores(
    recons: torch.Tensor,
    target: torch.Tensor,
    recons_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Get anomaly scores for a batch of jets.

    :param recons: Reconstructed jets.
    :type recons: torch.Tensor
    :param target: Target jets.
    :type target: torch.Tensor
    :param recons_normalized: Normalized reconstructed jets.
    :type recons_normalized: torch.Tensor
    :param target_normalized: Normalized target jets.
    :type target_normalized: torch.Tensor
    :return: A dictionary with the scores (value) for each type (key).
    :rtype: Dict[str, np.ndarray]
    """
    return {
        KEY_MSE: img_MSE(recons, target).detach().cpu().numpy(),
        KEY_MSE_NORM: img_MSE(recons_normalized, target_normalized)
        .detach()
        .cpu()
        .numpy(),
    }


def img_MSE(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).mean(dim=(-2, -1))
