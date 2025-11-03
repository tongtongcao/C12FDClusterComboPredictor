import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from event import extract_event

plt.rcParams.update({
    'font.size': 14,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth': 2,
    'lines.linewidth': 3
})

class Plotter:
    """
    Central plotting and statistics tool for particle track prediction.

    Parameters
    ----------
    print_dir : str
        Directory to save output figures.
    end_name : str
        Filename suffix for saved figures.
    """

    def __init__(self, print_dir='', end_name=''):
        self.print_dir = print_dir
        self.end_name = end_name
        if self.print_dir and not os.path.exists(self.print_dir):
            os.makedirs(self.print_dir)

    # -----------------------------
    # Training & Validation Loss
    # -----------------------------
    def plotTrainLoss(self, tracker):
        train_losses = tracker.train_losses
        val_losses = tracker.val_losses
        plt.figure(figsize=(8, 6))
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
    # Edge Probability Histogram
    # -----------------------------
    def plot_edge_probs(self, y_true, y_pred_prob, bins=20, title="Edge Probabilities"):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred_prob, torch.Tensor):
            y_pred_prob = y_pred_prob.cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.hist(y_pred_prob[y_true == 0], bins=bins, alpha=0.6, color='blue', label='Label 0')
        plt.hist(y_pred_prob[y_true == 1], bins=bins, alpha=0.6, color='red', label='Label 1')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        outname = f"{self.print_dir}/edge_probs_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    # -----------------------------
    # All Edges Plot
    # -----------------------------
    def plot_all_edges(self, predictor, data, title="All Edges", save_path=None):
        _, probs = predictor.predict_edges(data)
        edge_index = data.edge_index.t().tolist()
        edge_probs = probs.tolist()
        edge_labels = data.edge_label.tolist() if hasattr(data, "edge_label") else [0] * len(edge_index)

        avgWire = data.x[:, 0].cpu().numpy() * 112.0
        superlayer = data.x[:, 2].cpu().numpy() * 6.0

        plt.figure(figsize=(8, 6))
        plt.scatter(superlayer, avgWire, color='black', s=50, label='Nodes')
        for (i, j), p, l in zip(edge_index, edge_probs, edge_labels):
            x_coords = [superlayer[i], superlayer[j]]
            y_coords = [avgWire[i], avgWire[j]]
            plt.plot(x_coords, y_coords, color='gray', alpha=0.5)
            x_mid = sum(x_coords) / 2
            y_mid = sum(y_coords) / 2
            plt.text(x_mid, y_mid, f"{p:.2f}/{int(l)}", color='blue', fontsize=8, ha='center', va='bottom')

        plt.xlabel("Superlayer")
        plt.ylabel("avgWire")
        plt.ylim(0, 120)
        plt.title(title)
        plt.xticks(np.arange(1, 7))
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    # -----------------------------
    # Predicted Tracks
    # -----------------------------
    def plot_predicted_tracks(self, predictor, data, title="Predicted Tracks", save_path=None):
        tracks, track_probs, track_labels, noise_hits = predictor.predict_tracks(data)
        avgWire = data.x[:, 0].cpu().numpy() * 112.0
        superlayer = data.x[:, 2].cpu().numpy() * 6.0

        cluster_id_to_idx = {cid.item(): i for i, cid in enumerate(data.cluster_id)} if hasattr(data, "cluster_id") else None
        def map_track(tr):
            if cluster_id_to_idx:
                return [cluster_id_to_idx[cid] for cid in tr if cid in cluster_id_to_idx]
            return tr

        tracks_mapped = [map_track(tr) for tr in tracks]
        noise_hits_mapped = map_track(noise_hits) if noise_hits else None
        colors = plt.cm.get_cmap("tab10", 10)

        for t_idx, track in enumerate(tracks_mapped):
            if not track:
                continue
            plt.scatter(superlayer[track], avgWire[track], label=f"Track {t_idx+1}", s=50, color=colors(t_idx % 10))
            plt.plot(superlayer[track], avgWire[track], color=colors(t_idx % 10), linestyle='--', alpha=0.7)
            if track_probs and track_labels and t_idx < len(track_probs):
                for i in range(len(track)-1):
                    x_mid = (superlayer[track[i]] + superlayer[track[i+1]]) / 2
                    y_mid = (avgWire[track[i]] + avgWire[track[i+1]]) / 2
                    plt.text(x_mid, y_mid, f"{track_probs[t_idx][i]:.2f}/{track_labels[t_idx][i]}",
                             color=colors(t_idx % 10), fontsize=10, ha='center', va='bottom')

        if noise_hits_mapped:
            plt.scatter(superlayer[noise_hits_mapped], avgWire[noise_hits_mapped], label="Noise", s=50, color="gray", alpha=0.6)

        plt.xlabel("Superlayer")
        plt.ylabel("avgWire")
        plt.ylim(0, 120)
        plt.title(title)
        plt.xticks(np.arange(1, 7))
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    # -----------------------------
    # Precision / Recall / F1
    # -----------------------------
    def precision_recall_f1_with_best_threshold(self, y_true, y_pred_prob, thresholds=None, plot=True, method="f1_max"):
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)

        y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        y_pred_prob = y_pred_prob.cpu().numpy() if isinstance(y_pred_prob, torch.Tensor) else y_pred_prob

        precisions, recalls, f1s = [], [], []
        for thr in thresholds:
            y_pred = (y_pred_prob > thr).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))

        precisions, recalls, f1s = np.array(precisions), np.array(recalls), np.array(f1s)
        best_idx = np.argmax(f1s) if method == "f1_max" else np.argmin(np.abs(precisions - recalls))
        best_thr = thresholds[best_idx]

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precisions, label="Precision", marker='o')
            plt.plot(thresholds, recalls, label="Recall", marker='s')
            plt.plot(thresholds, f1s, label="F1", marker='^', linestyle='--', color='black')
            plt.scatter(thresholds[best_idx], f1s[best_idx], color='red', s=80,
                        label=f"Best {method}\nthr={best_thr:.2f}\nF1={f1s[best_idx]:.2f}")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title("Precision / Recall / F1 vs Threshold")
            plt.legend()
            plt.grid(True)
            out = f"{self.print_dir}/prf1_{self.end_name}.png"
            plt.savefig(out)
            plt.close()

        return precisions[best_idx], recalls[best_idx], f1s[best_idx], best_thr

    # -----------------------------
    # TPR / TNR vs Threshold
    # -----------------------------
    def tpr_tnr_vs_threshold(self, y_true, y_pred_prob, thresholds=None, plot=True):
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)

        y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        y_pred_prob = y_pred_prob.cpu().numpy() if isinstance(y_pred_prob, torch.Tensor) else y_pred_prob

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

    # -----------------------------
    # Track Purity / Efficiency per True Track
    # -----------------------------
    def track_purity_efficiency_per_true_track(self, predicted_tracks, data):
        trkIds_list = [set(t) for t in data.track_ids[0]]
        cluster_ids = data.cluster_id.tolist() if hasattr(data, "cluster_id") else list(range(1, len(trkIds_list)+1))
        cid2idx = {cid: i for i, cid in enumerate(cluster_ids)}

        predicted_node_tracks = [[cid2idx[cid] for cid in tr if cid in cid2idx] for tr in predicted_tracks]
        all_true_ids = {tid for s in trkIds_list for tid in s if tid != -1}
        results = {}

        for tid in all_true_ids:
            true_nodes = [i for i, s in enumerate(trkIds_list) if tid in s]
            matched_preds = [tr for tr in predicted_node_tracks if len(set(tr) & set(true_nodes)) > 0]
            if not matched_preds:
                results[tid] = {"purity": 0.0, "efficiency": 0.0,
                                "matched_pred": [], "true_cluster_ids": [cluster_ids[i] for i in true_nodes]}
                continue
            best_pred = max(matched_preds, key=lambda tr: len(set(tr) & set(true_nodes)))
            overlap = len(set(best_pred) & set(true_nodes))
            purity = overlap / len(best_pred)
            efficiency = overlap / len(true_nodes)
            results[tid] = {"purity": purity, "efficiency": efficiency,
                            "matched_pred": [cluster_ids[i] for i in best_pred],
                            "true_cluster_ids": [cluster_ids[i] for i in true_nodes]}
        return results

    # -----------------------------
    # Track Purity / Efficiency vs Threshold
    # -----------------------------
    def track_purity_efficiency_vs_threshold(self, predictor, data_loader, thresholds=None, plot=True):
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)

        avg_purity, avg_eff = [], []

        for thr in thresholds:
            purities_all, effs_all = [], []
            with torch.no_grad():
                for batch in data_loader:
                    for event_id in batch.batch.unique():
                        event_data = extract_event(batch, event_id)
                        tracks, _, _, _ = predictor.predict_tracks(event_data, threshold=thr)
                        metrics = self.track_purity_efficiency_per_true_track(tracks, event_data)
                        purities_all.extend([v["purity"] for v in metrics.values()])
                        effs_all.extend([v["efficiency"] for v in metrics.values()])
            avg_purity.append(np.mean(purities_all) if purities_all else 0.0)
            avg_eff.append(np.mean(effs_all) if effs_all else 0.0)

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, avg_purity, label="Avg Purity", marker='o')
            plt.plot(thresholds, avg_eff, label="Avg Efficiency", marker='s')
            plt.xlabel("Threshold")
            plt.ylabel("Value")
            plt.title("Track Purity & Efficiency vs Threshold")
            plt.legend()
            plt.grid(True)
            out = f"{self.print_dir}/track_purity_eff_vs_threshold_{self.end_name}.png"
            plt.savefig(out)
            plt.close()

        return {
            "thresholds": thresholds,
            "avg_purity": avg_purity,
            "avg_efficiency": avg_eff
        }
