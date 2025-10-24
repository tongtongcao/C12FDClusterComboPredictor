"""
Nodes = clusters; node features include avgWire, superlayer (one-hot encoded).

For each event, generate candidate edge pairs (i, j) (only within the same event), and assign labels as follows:
if the two clusters’ trkIds have an intersection and are not equal to -1, label as positive (1); otherwise, label as negative (0).

Construct a PyG Data object where

x is the node feature matrix,

edge_index represents the undirected candidate edge indices (or all candidate pairs),

edge_attr can be defined as [|Δsuperlayer|, |ΔavgWire|],

and y_edge contains the edge labels.

Use a GNN (e.g., GraphSAGE) to encode node embeddings, then concatenate features as
torch.cat([h_i, h_j, edge_attr])
and feed them into an MLP for binary edge classification.

During training, batch data by event (using a DataLoader), optimize with BCE or CE loss, and evaluate with AUC, precision, recall, and F1 metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import random
from pytorch_lightning.callbacks import Callback
from typing import List

# -----------------------------
# 解析 trkIds
# -----------------------------
def parse_trkids(trk_series: pd.Series) -> list[set[int]]:
    """
    将 pd.Series 中每个 trkIds 字符串解析为 set[int]。
    """
    trk_sets = []
    for trk_str in trk_series:
        if pd.isna(trk_str):
            trk_sets.append(set())
            continue
        s = str(trk_str).strip()
        if s == '' or s == '-1':
            trk_sets.append(set())
            continue
        try:
            trk_sets.append(set(int(t) for t in s.split(';') if t != '' and int(t) != -1))
        except Exception:
            trk_sets.append(set())
    return trk_sets


# -----------------------------
# 向量化构建图（含边特征 & 标签）
# -----------------------------
def build_graph_from_hits(
    clusters: pd.DataFrame,
    keep_all_noise_prob: float = 0.1,
    avgWire_diff_max: list[list[float]] = None,
    bidirectional: bool = True
):
    if avgWire_diff_max is None:
        avgWire_diff_max = [
            [12.0, 20.0, 12.0, 38.0, 14.0],
            [14.0, 18.0, 40.0, 40.0],
            [24.0, 48.0, 48.0],
            [55.0, 55.0],
            [56.0],
        ]

    num_clusters = len(clusters)
    if num_clusters == 0:
        return None

    avgWire = clusters["avgWire"].values.astype(np.float32)
    superlayer = clusters["superlayer"].values.astype(np.int32)
    cluster_ids = clusters["clusterIdx"].values.astype(np.int32)
    track_ids_list = parse_trkids(clusters["trkIds"])

    # -----------------------------
    # 节点特征归一化
    # -----------------------------
    wire_min, wire_max = avgWire.min(), avgWire.max()

    wire_range = 112.0
    superlayer_range = 6.0

    avgWire_norm = (avgWire - wire_min) / wire_range
    superlayer_norm = superlayer / superlayer_range

    x = torch.tensor(np.stack([avgWire_norm, superlayer_norm], axis=1), dtype=torch.float)

    # -----------------------------
    # 边计算向量化
    # -----------------------------
    L1 = superlayer[:, None]
    L2 = superlayer[None, :]
    dL = np.abs(L2 - L1)

    W1 = avgWire[:, None]
    W2 = avgWire[None, :]
    diff = W2 - W1  # 保留符号用于 edge_attr

    mask = np.zeros((num_clusters, num_clusters), dtype=bool)
    for delta in range(1, len(avgWire_diff_max)+1):
        idx = np.where(dL == delta)
        for i, j in zip(*idx):
            lower = min(superlayer[i], superlayer[j])
            if lower - 1 >= len(avgWire_diff_max[delta-1]):
                continue
            max_diff = avgWire_diff_max[delta-1][lower-1]
            if abs(diff[i,j]) <= max_diff:
                mask[i,j] = True

    # 获取 i<j 的边索引
    src, dst = np.where(np.triu(mask, k=1))  # 上三角，避免 self-loop

    if len(src) == 0:
        return None

    # 边特征
    sl_diff = superlayer[src] - superlayer[dst]
    aw_diff = avgWire[src] - avgWire[dst]
    edge_attr = np.stack([sl_diff / superlayer_range, aw_diff / wire_range], axis=1).astype(np.float32)

    # 边标签：1=共轨迹，0=无交集
    edge_label = np.array([1 if len(track_ids_list[i] & track_ids_list[j]) > 0 else 0
                           for i, j in zip(src, dst)], dtype=np.float32)

    # -----------------------------
    # 保留部分纯噪声事件
    # -----------------------------
    if edge_label.sum() == 0 and random.random() > keep_all_noise_prob:
        return None

    # -----------------------------
    # 双向边处理
    # -----------------------------
    if bidirectional:
        src_all = np.concatenate([src, dst])
        dst_all = np.concatenate([dst, src])
        edge_index = torch.tensor(np.stack([src_all, dst_all], axis=0), dtype=torch.long)
        edge_attr = torch.tensor(np.vstack([edge_attr, -edge_attr]), dtype=torch.float)
        edge_label = torch.tensor(np.concatenate([edge_label, edge_label]), dtype=torch.float)
    else:
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_label = torch.tensor(edge_label, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        superlayer=torch.tensor(superlayer, dtype=torch.long),
        track_ids=[list(s) for s in track_ids_list],
        cluster_id=torch.tensor(cluster_ids, dtype=torch.long)
    )
    return data

# -----------------------------
# EdgeClassifier: supports edge features & pos_weight BCE
# -----------------------------
class EdgeClassifier(pl.LightningModule):
    """
    GraphSAGE-based edge classifier with optional edge features.

    Parameters
    ----------
    in_channels : int
        Node feature dimension.
    hidden_channels : int
        Hidden dimension of GraphSAGE and MLP.
    num_layers : int
        Number of GraphSAGE layers.
    lr : float
        Learning rate.
    dropout : float
        Dropout probability in MLP and GNN layers.
    """
    def __init__(self, in_channels=2, hidden_channels=32, num_layers=2, lr=1e-3, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_ch, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # MLP input: [h_i || h_j || edge_attr]
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels + 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        self.lr = lr
        self.dropout = dropout

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : Batch
            PyG Batch or single Data object containing x, edge_index, edge_attr.

        Returns
        -------
        torch.Tensor
            Edge logits of shape (num_edges,)
        """
        if not isinstance(data, Batch):
            data = Batch.from_data_list([data])

        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        src, dst = edge_index
        if edge_attr is None:
            device = x.device
            edge_attr = torch.zeros((src.size(0), 2), dtype=x.dtype, device=device)

        edge_attr = edge_attr.to(x.device).to(x.dtype)
        edge_feat = torch.cat([x[src], x[dst], edge_attr], dim=1)
        logits = self.classifier(edge_feat).view(-1)
        return logits

    def training_step(self, batch: Batch) -> torch.Tensor:
        """
        Training step for Lightning.

        Returns
        -------
        torch.Tensor
            BCE loss
        """
        logits = self(batch)
        labels = batch.edge_label.float()

        pos_count = (labels == 1.0).sum().item()
        neg_count = (labels == 0.0).sum().item()
        pos_weight = torch.tensor(max(neg_count / max(1.0, pos_count), 1.0), dtype=torch.float, device=logits.device)

        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch: Batch) -> torch.Tensor:
        """
        Validation step for Lightning.

        Returns
        -------
        torch.Tensor
            BCE loss
        """
        logits = self(batch)
        labels = batch.edge_label.float()
        pos_count = (labels == 1.0).sum().item()
        neg_count = (labels == 0.0).sum().item()
        pos_weight = torch.tensor(max(neg_count / max(1.0, pos_count), 1.0), dtype=torch.float, device=logits.device)

        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# -----------------------------
# Inference Wrapper: single or batch graph
# -----------------------------
class EdgeClassifierWrapper(nn.Module):
    """
    TorchScript-compatible EdgeClassifier wrapper.
    Only Tensor fields are used; supports arbitrary node/edge counts.

    Forward signature:
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_feat_dim] or empty tensor
    Returns:
        logits: [num_edges]
    """

    def __init__(self, model: nn.Module, edge_feat_dim: int = 2):
        super().__init__()
        # Copy GNN layers
        self.convs = model.convs
        self.bns = model.bns
        self.classifier = model.classifier
        self.dropout = getattr(model, "dropout", 0.1)
        self.edge_feat_dim = edge_feat_dim

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # Ensure edge_attr is tensor
        if edge_attr.numel() == 0:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_feat_dim),
                                    dtype=x.dtype, device=x.device)

        # GraphSAGE forward
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Gather source and target node embeddings
        src = edge_index[0]
        dst = edge_index[1]

        # Concatenate node embeddings and edge features
        edge_feat = torch.cat([x[src], x[dst], edge_attr], dim=1)

        # MLP classifier
        logits = self.classifier(edge_feat).view(-1)
        return logits

# -----------------------------
# Callback to track training & validation loss
# -----------------------------
class LossTracker(Callback):
    """
    PyTorch Lightning callback to track training and validation losses.
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics["val_loss"].item())




