import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

plt.rcParams.update({
    'font.size': 14,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth': 2,
    'lines.linewidth': 3
})

class Plotter:
    def __init__(self, print_dir='', end_name=''):
        """
        Plotter 类：集中绘图与统计分析工具

        Parameters
        ----------
        print_dir : str
            输出图像目录
        end_name : str
            文件名后缀
        """
        self.print_dir = print_dir
        self.end_name = end_name
        if self.print_dir and not os.path.exists(self.print_dir):
            os.makedirs(self.print_dir)

    # -----------------------------
    # 绘制训练和验证 Loss 曲线
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

    def plot_edge_probs(self, y_true, y_pred_prob, bins=20, title="Edge Probabilities"):
        """
        绘制所有边的预测概率分布，同时区分真实标签为0和1的边。

        Parameters
        ----------
        y_true : np.ndarray or torch.Tensor
            真实边标签，0或1
        y_pred_prob : np.ndarray or torch.Tensor
            预测边概率，0~1
        bins : int
            直方图的bin数量
        title : str
            图标题
        save_path : str, optional
            保存路径
        """
        
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

    def plot_predicted_tracks(self, data, tracks, track_probs=None, noise_hits=None, title="Predicted Tracks",
                              save_path=None):
        plt.figure(figsize=(8, 6))
        avgWire = data.x[:, 0].cpu().numpy()
        if data.x.size(1) > 1:
            superlayer = data.x[:, 1:7].argmax(dim=1).cpu().numpy() + 1
        else:
            superlayer = data.superlayer.cpu().numpy()

        cluster_id_to_idx = None
        if hasattr(data, "cluster_id"):
            cluster_id_to_idx = {cid.item(): i for i, cid in enumerate(data.cluster_id)}

        def map_track(tr):
            if cluster_id_to_idx is not None:
                return [cluster_id_to_idx[cid] for cid in tr if cid in cluster_id_to_idx]
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
            plt.scatter(superlayer[track], avgWire[track], label=f"Track {t_idx + 1}", s=50, color=colors(t_idx % 10))
            plt.plot(superlayer[track], avgWire[track], color=colors(t_idx % 10), linestyle='--', alpha=0.7)

            # -----------------------------
            # 在每条边上显示概率
            # -----------------------------
            if track_probs is not None and t_idx < len(track_probs):
                for i in range(len(track) - 1):
                    x_mid = (superlayer[track[i]] + superlayer[track[i + 1]]) / 2
                    y_mid = (avgWire[track[i]] + avgWire[track[i + 1]]) / 2
                    plt.text(x_mid, y_mid, f"{track_probs[t_idx][i]:.2f}", color=colors(t_idx % 10),
                             fontsize=10, ha='center', va='bottom')

        if noise_hits_mapped is not None and len(noise_hits_mapped) > 0:
            plt.scatter(superlayer[noise_hits_mapped], avgWire[noise_hits_mapped],
                        label="Noise", s=50, color="gray", alpha=0.6)

        plt.xlabel("Superlayer")
        plt.ylabel("avgWireNorm")
        plt.title(title)
        plt.xticks(np.arange(1, 7))
        plt.legend()
        plt.grid(True)
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()

    # -----------------------------
    # Precision / Recall / F1 vs 阈值
    # -----------------------------
    def precision_recall_f1_with_best_threshold(self, y_true, y_pred_prob,
                                                thresholds=None, plot=True,
                                                method="f1_max"):
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
            plt.scatter(thresholds[best_idx], f1s[best_idx],
                        color='red', s=80,
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

    # -----------------------------
    # 每条真实轨迹的纯度和效率
    # -----------------------------
    def track_purity_efficiency_per_true_track(self, predicted_tracks, data):
        """
        计算每条真实轨迹的 purity & efficiency
        支持 data.trkIds 为 list of sets（允许空集、多id）
        """
        trkIds_list = data.trkIds
        all_true_ids = set()
        for s in trkIds_list:
            all_true_ids.update(s)
        if -1 in all_true_ids:
            all_true_ids.remove(-1)

        result = {}
        for tid in all_true_ids:
            true_nodes = [i for i, s in enumerate(trkIds_list) if tid in s]
            matched_pred = [tr for tr in predicted_tracks if len(set(tr) & set(true_nodes)) > 0]
            pred_nodes_union = set().union(*matched_pred) if matched_pred else set()

            purity = len(set(true_nodes) & pred_nodes_union) / len(pred_nodes_union) if pred_nodes_union else 0.0
            efficiency = len(set(true_nodes) & pred_nodes_union) / len(true_nodes) if true_nodes else 0.0
            result[tid] = {"purity": purity, "efficiency": efficiency}
        return result

    # -----------------------------
    # 不同 threshold 下的轨迹效率 / 纯度 曲线
    # -----------------------------
    def track_purity_efficiency_vs_threshold(self, predictor, data,
                                             thresholds=None, plot=True):
        """
        利用 predictor.predict_tracks，在不同 threshold 下计算平均 purity/efficiency
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)

        avg_purity, avg_eff = [], []

        for thr in thresholds:
            pred_tracks = predictor.predict_tracks(data, threshold=thr)
            metrics = self.track_purity_efficiency_per_true_track(pred_tracks, data)
            purities = [v["purity"] for v in metrics.values()]
            effs = [v["efficiency"] for v in metrics.values()]
            avg_purity.append(np.mean(purities) if purities else 0.0)
            avg_eff.append(np.mean(effs) if effs else 0.0)

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
