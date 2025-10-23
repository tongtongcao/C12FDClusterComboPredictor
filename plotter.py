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

    def plot_all_edges(self, predictor, data, title="All Edges", save_path=None):
        """
        绘制整图所有边，并显示预测概率和真实 label。

        Parameters
        ----------
        predictor : TrackPredictor
            用于预测边概率。
        data : torch_geometric.data.Data
            图数据，必须包含 x, superlayer, 以及 edge_label（真实0/1标签）。
        title : str
            图标题。
        save_path : str, optional
            保存路径。
        """
        # 获取预测概率
        _, probs = predictor.predict_edges(data)
        edge_index = data.edge_index.t().tolist()
        edge_probs = probs.tolist()

        # 真实 label
        if hasattr(data, "edge_label"):
            edge_labels = data.edge_label.tolist()
        else:
            edge_labels = [0] * len(edge_index)  # 默认全 0

        num_nodes = data.x.size(0)
        avgWire = data.x[:, 0].cpu().numpy()
        superlayer = data.x[:, 1].cpu().numpy() * 6.0

        plt.figure(figsize=(8, 6))
        plt.scatter(superlayer, avgWire, color='black', s=50, label='Nodes')

        for (i, j), p, l in zip(edge_index, edge_probs, edge_labels):
            x_coords = [superlayer[i], superlayer[j]]
            y_coords = [avgWire[i], avgWire[j]]
            plt.plot(x_coords, y_coords, color='gray', alpha=0.5)
            x_mid = sum(x_coords) / 2
            y_mid = sum(y_coords) / 2
            plt.text(x_mid, y_mid, f"{p:.2f}/{l}", color='blue', fontsize=8, ha='center', va='bottom')

        plt.xlabel("Superlayer")
        plt.ylabel("avgWireNorm")
        plt.title(title)
        plt.xticks(np.arange(1, 7))
        plt.grid(True)
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()

    def plot_predicted_tracks(self, predictor, data, title="Predicted Tracks", save_path=None):
        """
        绘制预测轨迹，每条轨迹显示每条边的预测概率和真实 label。

        Parameters
        ----------
        data : torch_geometric.data.Data
            图数据，包含 x (节点特征) 和 superlayer。
        tracks : list of list of int
            每条轨迹的节点 cluster_id。
        track_probs : list of list of float, optional
            每条轨迹每条边的预测概率。
        track_labels : list of list of int, optional
            每条轨迹每条边的真实 label（0/1）。
        noise_hits : list of int, optional
            噪声节点 cluster_id。
        title : str
            图标题。
        save_path : str, optional
            保存路径。
        """

        tracks, track_probs, track_labels, noise_hits = predictor.predict_tracks(data)

        plt.figure(figsize=(8, 6))
        avgWire = data.x[:, 0].cpu().numpy()
        superlayer = data.x[:, 1].cpu().numpy() * 6.0

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
            plt.scatter(superlayer[track], avgWire[track],
                        label=f"Track {t_idx + 1}", s=50, color=colors(t_idx % 10))
            plt.plot(superlayer[track], avgWire[track], color=colors(t_idx % 10),
                     linestyle='--', alpha=0.7)

            # -----------------------------
            # 显示每条边的预测概率和真实 label
            # -----------------------------
            if track_probs is not None and track_labels is not None and t_idx < len(track_probs):
                for i in range(len(track) - 1):
                    x_mid = (superlayer[track[i]] + superlayer[track[i + 1]]) / 2
                    y_mid = (avgWire[track[i]] + avgWire[track[i + 1]]) / 2
                    plt.text(
                        x_mid, y_mid,
                        f"{track_probs[t_idx][i]:.2f}/{track_labels[t_idx][i]}",
                        color=colors(t_idx % 10),
                        fontsize=10, ha='center', va='bottom'
                    )

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
    # 真实轨迹的纯度和效率
    # -----------------------------
    def track_purity_efficiency_per_true_track(self, predicted_tracks, data):
        """
        计算每条真实轨迹的 purity 和 efficiency。
        支持 predicted_tracks 和 true_tracks 都用 cluster_id 表示（从1开始）。

        Parameters
        ----------
        predicted_tracks : list of list
            预测轨迹，每条轨迹是 cluster_id 的列表（1-based）
        data : torch_geometric.data.Data
            包含 data.track_ids（每个节点对应真实轨迹ID列表）和 data.cluster_id（节点对应的 cluster_id）

        Returns
        -------
        dict
            key: 真实轨迹ID (tid)
            value: dict {
                "purity": float,
                "efficiency": float,
                "matched_pred": list of cluster_id (1-based),
                "true_cluster_ids": list of cluster_id (1-based)
            }
        """
        # 1️⃣ 提取真实轨迹列表并转为 set（每个节点可能属于多个真实轨迹）
        trkIds_list = [set(t) for t in data.track_ids[0]]

        # 2️⃣ 建立 cluster_id -> 节点索引 映射
        if hasattr(data, "cluster_id"):
            cluster_ids = data.cluster_id.tolist()
        else:
            # 如果没有 cluster_id，则假设 cluster_id = node_index + 1
            cluster_ids = list(range(1, len(trkIds_list) + 1))
        cid2idx = {cid: i for i, cid in enumerate(cluster_ids)}

        # 3️⃣ 把 predicted_tracks 的 cluster_id 转换为节点索引（0-based）
        predicted_node_tracks = [
            [cid2idx[cid] for cid in tr if cid in cid2idx]
            for tr in predicted_tracks
        ]

        # 4️⃣ 获取所有真实轨迹ID
        all_true_ids = {tid for s in trkIds_list for tid in s if tid != -1}

        results = {}

        # 5️⃣ 遍历每条真实轨迹
        for tid in all_true_ids:
            # 找出属于该真轨迹的节点索引
            true_nodes = [i for i, s in enumerate(trkIds_list) if tid in s]
            if not true_nodes:
                continue

            # 找出与该真轨迹有交集的预测轨迹
            matched_preds = [
                tr for tr in predicted_node_tracks if len(set(tr) & set(true_nodes)) > 0
            ]

            # 如果没有匹配轨迹，则 purity/efficiency 都为0
            if not matched_preds:
                results[tid] = {
                    "purity": 0.0,
                    "efficiency": 0.0,
                    "matched_pred": [],
                    "true_cluster_ids": [cluster_ids[i] for i in true_nodes]
                }
                continue

            # 选出与真轨迹交集最多的预测轨迹
            best_pred = max(matched_preds, key=lambda tr: len(set(tr) & set(true_nodes)))
            overlap = len(set(best_pred) & set(true_nodes))

            purity = overlap / len(best_pred)
            efficiency = overlap / len(true_nodes)

            results[tid] = {
                "purity": purity,
                "efficiency": efficiency,
                "matched_pred": [cluster_ids[i] for i in best_pred],  # 转回 cluster_id
                "true_cluster_ids": [cluster_ids[i] for i in true_nodes]  # 转回 cluster_id
            }

        return results

    # -----------------------------
    # 不同 threshold 下的轨迹效率 / 纯度 曲线
    # -----------------------------
    def track_purity_efficiency_vs_threshold(self, predictor, data_loader,
                                             thresholds=None, plot=True):
        """
        利用 predictor.predict_tracks，在不同 threshold 下计算平均 purity/efficiency
        支持 data_loader 输入 (torch_geometric DataLoader)

        Parameters
        ----------
        predictor : TrackPredictor
            用于预测轨迹的 TrackPredictor 对象
        data_loader : DataLoader
            验证集 DataLoader
        thresholds : list or np.ndarray, optional
            阈值列表
        plot : bool
            是否绘图

        Returns
        -------
        dict
            {
                "thresholds": thresholds,
                "avg_purity": list of float,
                "avg_efficiency": list of float
            }
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)

        avg_purity, avg_eff = [], []

        for thr in thresholds:
            purities_all, effs_all = [], []

            with torch.no_grad():
                for batch in data_loader:
                    # 遍历 batch 中每个事件
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

