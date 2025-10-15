import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

plt.rcParams.update({
    'font.size': 15,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 15,
    'xtick.minor.size': 10,
    'ytick.major.size': 15,
    'ytick.minor.size': 10,
    'xtick.major.width': 3,
    'xtick.minor.width': 3,
    'ytick.major.width': 3,
    'ytick.minor.width': 3,
    'axes.linewidth': 3,
    'figure.max_open_warning': 200,
    'lines.linewidth': 5
})


class Plotter:
    def __init__(self, print_dir='', end_name=''):
        """
        Initialize Plotter.

        Parameters:
        -----------
        print_dir : str
            Directory to save plots.
        end_name : str
            Suffix to append to saved filenames.
        """
        self.print_dir = print_dir
        self.end_name = end_name
        if self.print_dir and not os.path.exists(self.print_dir):
            os.makedirs(self.print_dir)

    # -----------------------------
    # Plot training/validation loss
    # -----------------------------
    def plotTrainLoss(self, tracker):
        """
        Plot training and validation loss curves.

        Parameters:
        -----------
        tracker : object
            Object with attributes `train_losses` and `val_losses`.
        """
        train_losses = tracker.train_losses
        val_losses = tracker.val_losses
        plt.figure(figsize=(20, 20))
        plt.plot(train_losses, label='Train', color='royalblue')
        plt.plot(val_losses, label='Validation', color='firebrick')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        outname = f"{self.print_dir}/loss_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    # -----------------------------
    # Visualize predicted tracks
    # -----------------------------
    def plot_predicted_tracks(self, data, tracks, noise_hits=None, title="Predicted Tracks", save_path=None):
        """
        Plot predicted tracks with avgWire vs superlayer.

        Supports tracks given as node indices or cluster_id. If cluster_id is used,
        data.cluster_id must exist to map back to node indices.

        Parameters:
        -----------
        data : object
            Data object containing node features `x` and optionally `superlayer`.
            If using cluster_id, data.cluster_id must exist.
        tracks : list of list
            List of predicted tracks, each a list of node indices or cluster_ids.
        noise_hits : list, optional
            List of noise node indices or cluster_ids.
        title : str, optional
            Plot title.
        save_path : str, optional
            Path to save the figure. If None, figure is not saved.
        """
        plt.figure(figsize=(8, 6))
        avgWire = data.x[:, 0].cpu().numpy()
        if data.x.size(1) > 1:
            superlayer = data.x[:, 1:7].argmax(dim=1).cpu().numpy() + 1
        else:
            superlayer = data.superlayer.cpu().numpy()

        # -----------------------------
        # 如果是 cluster_id，映射回节点索引
        # -----------------------------
        cluster_id_to_idx = None
        if hasattr(data, "cluster_id"):
            cluster_id_to_idx = {cid.item(): i for i, cid in enumerate(data.cluster_id)}

        def map_track(tr):
            if cluster_id_to_idx is not None:
                mapped = []
                for cid in tr:
                    if cid in cluster_id_to_idx:
                        mapped.append(cluster_id_to_idx[cid])
                    else:
                        print(f"Warning: cluster_id {cid} not found in event_data.cluster_id")
                return mapped
            return tr

        tracks_mapped = [map_track(tr) for tr in tracks]
        if noise_hits is not None:
            noise_hits_mapped = map_track(noise_hits)
        else:
            noise_hits_mapped = None

        colors = plt.cm.get_cmap("tab10", 10)
        for t_idx, track in enumerate(tracks_mapped):
            if len(track) == 0:
                continue
            plt.scatter(superlayer[track], avgWire[track], label=f"Track {t_idx + 1}",
                        s=50, color=colors(t_idx % 10))
            plt.plot(superlayer[track], avgWire[track], color=colors(t_idx % 10),
                     linestyle='--', alpha=0.7)

        if noise_hits_mapped is not None and len(noise_hits_mapped) > 0:
            plt.scatter(superlayer[noise_hits_mapped], avgWire[noise_hits_mapped],
                        label="Noise", s=50, color="gray", alpha=0.6)

        plt.xlabel("Superlayer")
        plt.ylabel("avgWire")
        plt.title(title)
        plt.xticks(np.arange(1, 7))
        plt.legend()
        plt.grid(True)
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()

    # -----------------------------
    # Automatically select best threshold
    # -----------------------------
    def precision_recall_f1_with_best_threshold(self, y_true, y_pred_prob, thresholds=None, plot=True, method="f1_max"):
        """
        Automatically select the best threshold based on F1 score or precision-recall balance.

        Parameters:
        -----------
        y_true : torch.Tensor or np.ndarray
            True edge labels (0/1).
        y_pred_prob : torch.Tensor or np.ndarray
            Predicted edge probabilities (0~1).
        thresholds : list or np.ndarray, optional
            List of thresholds to evaluate. Default: 21 points between 0 and 1.
        plot : bool
            Whether to plot Precision/Recall/F1 vs threshold.
        method : str
            Method to select threshold: 'f1_max' or 'precision_recall_balance'.

        Returns:
        --------
        precision_best : float
        recall_best : float
        f1_best : float
        best_threshold : float
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred_prob, torch.Tensor):
            y_pred_prob = y_pred_prob.cpu().numpy()

        precisions, recalls, f1_scores = [], [], []

        for thr in thresholds:
            y_pred = (y_pred_prob > thr).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1_scores = np.array(f1_scores)

        if method == "f1_max":
            best_idx = np.argmax(f1_scores)
        elif method == "precision_recall_balance":
            best_idx = np.argmin(np.abs(precisions - recalls))
        else:
            raise ValueError("method must be 'f1_max' or 'precision_recall_balance'")

        best_threshold = thresholds[best_idx]

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precisions, label="Precision", marker='o')
            plt.plot(thresholds, recalls, label="Recall", marker='s')
            plt.plot(thresholds, f1_scores, label="F1-score", marker='^', linestyle='--', color='black')
            plt.scatter(thresholds[best_idx], f1_scores[best_idx], color='red', s=100,
                        label=f"Best ({method})\nThr={best_threshold:.2f}\nF1={f1_scores[best_idx]:.2f}")
            plt.xlabel("Edge probability threshold")
            plt.ylabel("Score")
            plt.title("Precision / Recall / F1 vs Threshold")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            outname = f"{self.print_dir}/precision_recall_f1_{self.end_name}.png"
            plt.savefig(outname)
            plt.close()

        precision_best = precisions[best_idx]
        recall_best = recalls[best_idx]
        f1_best = f1_scores[best_idx]

        print(f"Selected threshold ({method}): {best_threshold:.2f}")
        print(f"Precision={precision_best:.4f}, Recall={recall_best:.4f}, F1={f1_best:.4f}")

        return precision_best, recall_best, f1_best, best_threshold

    # -----------------------------
    # TPR / TNR vs threshold
    # -----------------------------
    def tpr_tnr_vs_threshold(self, y_true, y_pred_prob, thresholds=None, plot=True):
        """
        Compute True Positive Rate (TPR) and True Negative Rate (TNR) vs threshold.

        Parameters:
        -----------
        y_true : torch.Tensor or np.ndarray
            True edge labels (0/1).
        y_pred_prob : torch.Tensor or np.ndarray
            Predicted edge probabilities (0~1).
        thresholds : list or np.ndarray, optional
            Threshold values to evaluate.
        plot : bool
            Whether to plot the TPR/TNR curves.

        Returns:
        --------
        tpr_list : np.ndarray
            TPR at each threshold.
        tnr_list : np.ndarray
            TNR at each threshold.
        thresholds : np.ndarray
            Evaluated thresholds.
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred_prob, torch.Tensor):
            y_pred_prob = y_pred_prob.cpu().numpy()

        tpr_list, tnr_list = [], []

        for thr in thresholds:
            y_pred = (y_pred_prob > thr).astype(int)
            TP = ((y_true == 1) & (y_pred == 1)).sum()
            FN = ((y_true == 1) & (y_pred == 0)).sum()
            TN = ((y_true == 0) & (y_pred == 0)).sum()
            FP = ((y_true == 0) & (y_pred == 1)).sum()

            TPR = TP / (TP + FN + 1e-8)
            TNR = TN / (TN + FP + 1e-8)

            tpr_list.append(TPR)
            tnr_list.append(TNR)

        tpr_list = np.array(tpr_list)
        tnr_list = np.array(tnr_list)

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, tpr_list, label="TPR", marker='o')
            plt.plot(thresholds, tnr_list, label="TNR", marker='s')
            plt.xlabel("Threshold")
            plt.ylabel("Rate")
            plt.title("TPR and TNR vs Threshold")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            outname = f"{self.print_dir}/tpr_tnr_{self.end_name}.png"
            plt.savefig(outname)
            plt.close()

        return tpr_list, tnr_list, thresholds
