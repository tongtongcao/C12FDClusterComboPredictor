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

        Parameters:
        -----------
        data : object
            Data object containing node features `x` and optionally `superlayer`.
        tracks : list of list
            List of predicted tracks, each a list of node indices.
        noise_hits : list, optional
            List of noise node indices.
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
        colors = plt.cm.get_cmap("tab10", 10)
        for t_idx, track in enumerate(tracks):
            plt.scatter(superlayer[track], avgWire[track], label=f"Track {t_idx + 1}", s=50, color=colors(t_idx % 10))
            plt.plot(superlayer[track], avgWire[track], color=colors(t_idx % 10), linestyle='--', alpha=0.7)
        if noise_hits is not None and len(noise_hits) > 0:
            plt.scatter(superlayer[noise_hits], avgWire[noise_hits], label="Noise", s=50, color="gray", alpha=0.6)
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
            plt.plot(thresholds, tpr_list, label="TPR (Recall)", marker='o')
            plt.plot(thresholds, tnr_list, label="TNR (Specificity)", marker='s')
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

    # -----------------------------
    # Track metrics
    # -----------------------------
    def compute_track_metrics(self, tracks_pred, tracks_true):
        """
        Compute track Purity (Precision) and Efficiency (Recall).

        Parameters:
        -----------
        tracks_pred : list of list
            Predicted tracks, each a list of node indices.
        tracks_true : list of list
            True tracks, each a list of node indices.

        Returns:
        --------
        purity : float
            Mean precision of predicted tracks.
        efficiency : float
            Recall (fraction of true hits recovered).
        """
        true_hits = set([hit for tr in tracks_true for hit in tr])
        pred_hits = set([hit for tr in tracks_pred for hit in tr])

        correct_hits = pred_hits & true_hits
        efficiency = len(correct_hits) / max(len(true_hits), 1)

        purities = []
        for tr in tracks_pred:
            tr_set = set(tr)
            correct = tr_set & true_hits
            #print(len(tr_set), len(true_hits), len(correct), max(len(tr_set), 1))
            purities.append(len(correct) / max(len(tr_set), 1))
        purity = np.mean(purities) if purities else 0.0

        return purity, efficiency

    # -----------------------------
    # Plot track metrics vs threshold
    # -----------------------------
    def plot_track_metrics_vs_threshold(self, track_predictor, val_loader, thresholds=None):
        """
        Predict tracks using TrackPredictor and plot Purity and Efficiency vs edge probability threshold.

        Parameters:
        -----------
        track_predictor : object
            TrackPredictor with a .predict_tracks() method and .threshold attribute.
        val_loader : iterable
            Validation data loader yielding batches.
        thresholds : list or np.ndarray, optional
            Threshold values to evaluate. Default 21 points between 0 and 1.
        min_track_length : int
            Only consider tracks with at least this many nodes.

        Returns:
        --------
        thresholds : np.ndarray
            Threshold values used.
        purity_list : list
            Mean Purity for each threshold.
        efficiency_list : list
            Mean Efficiency for each threshold.
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)

        purity_list, efficiency_list = [], []

        for thr in thresholds:
            track_predictor.threshold = thr
            purities, efficiencies = [], []

            with torch.no_grad():
                for batch in val_loader:
                    for event_id in batch.batch.unique():
                        node_mask = (batch.batch == event_id)

                        # -----------------------------
                        # 构建 event_data，并保证 track_ids 是列表的列表
                        # -----------------------------
                        edge_mask = None
                        if hasattr(batch, "edge_index"):
                            edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]

                        # 构造 track_ids，展开可能的嵌套列表
                        track_ids = []
                        for tid, m in zip(batch.track_ids, node_mask):
                            if not m:
                                continue
                            if tid is None or tid == -1:
                                track_ids.append([])
                            elif isinstance(tid, int):
                                track_ids.append([tid])
                            elif isinstance(tid, list):
                                flat = []
                                for t in tid:
                                    if isinstance(t, list):
                                        flat.extend(t)  # 展开嵌套列表
                                    else:
                                        flat.append(t)
                                track_ids.append(flat)
                            else:
                                track_ids.append([])

                        # 构建 event_data
                        event_data = type(batch)(
                            x=batch.x[node_mask],
                            superlayer=batch.superlayer[node_mask],
                            edge_index=batch.edge_index[:, edge_mask] if edge_mask is not None else None,
                            edge_label=batch.edge_label[edge_mask] if hasattr(batch,
                                                                              "edge_label") and edge_mask is not None else None,
                            edge_attr=batch.edge_attr[edge_mask] if hasattr(batch,
                                                                            "edge_attr") and edge_mask is not None else None,
                            track_ids=track_ids
                        )

                        # -----------------------------
                        # 预测轨迹
                        tracks_pred, _ = track_predictor.predict_tracks(event_data)

                        #print(event_data.track_ids)
                        # -----------------------------
                        # 构造 tracks_true
                        tracks_true_dict = {}
                        for idx, trk_entry in enumerate(event_data.track_ids):
                            for trk_id in trk_entry:
                                if trk_id == -1:
                                    continue
                                tracks_true_dict.setdefault(int(trk_id), []).append(idx)

                        tracks_true = list(tracks_true_dict.values())
                        #print(len(tracks_true), tracks_true)

                        purity, efficiency = self.compute_track_metrics(tracks_pred, tracks_true)
                        purities.append(purity)
                        efficiencies.append(efficiency)

            purity_list.append(np.mean(purities))
            efficiency_list.append(np.mean(efficiencies))

        # -----------------------------
        # 绘图
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, purity_list, label="Purity / Precision", marker='o')
        plt.plot(thresholds, efficiency_list, label="Efficiency / Recall", marker='s')
        plt.xlabel("Edge probability threshold")
        plt.ylabel("Metric")
        plt.title("Track Purity and Efficiency vs Threshold")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        outname = f"{self.print_dir}/metrics_vs_threshold_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

        return thresholds, purity_list, efficiency_list
