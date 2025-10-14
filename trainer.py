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

# -----------------------------
# Parse trkIds
# -----------------------------
def parse_trkids(trk_str: str) -> set[int]:
    """
    Parse a track ID string into a set of integers.

    Parameters
    ----------
    trk_str : str
        Track ID string, can be '-1', empty, or multiple IDs separated by ';'.

    Returns
    -------
    set[int]
        A set of track IDs. Empty set for invalid or missing IDs.
    """
    if pd.isna(trk_str):
        return set()
    s = str(trk_str).strip()
    if s == '' or s == '-1':
        return set()
    try:
        return set(int(t) for t in s.split(';') if t != '' and int(t) != -1)
    except Exception:
        return set()

# -----------------------------
# Build graph (with edge features & multi-track labels)
# -----------------------------
def build_graph_from_hits(
    clusters: pd.DataFrame,
    keep_all_noise_prob: float = 0.1,
    avgWire_diff_max=[[12.0, 20.0, 12.0, 38.0, 14.0], [14.0, 18.0, 40.0, 40.0]],
    bidirectional: bool = True
) -> Data | None:
    """
    Build a PyG Data object from cluster hits.

    Parameters
    ----------
    clusters : pd.DataFrame
        DataFrame containing columns: 'avgWire', 'superlayer', 'trkIds'.
    keep_all_noise_prob : float
        Probability to keep purely noisy events (no positive edges).
    avgWire_diff_max : list[list[float]]
        Thresholds for maximum allowed ΔavgWire between superlayers.
        Format: [[for adjacent layers], [for skip layers]]
    bidirectional : bool
        Whether to add edges in both directions.

    Returns
    -------
    Data | None
        PyG Data object with attributes:
        - x: node features (avgWire_norm + superlayer one-hot)
        - edge_index: candidate edge indices
        - edge_attr: edge features [Δsuperlayer, ΔavgWire_norm]
        - edge_label: edge labels (1 if tracks intersect, else 0)
        - superlayer: original superlayer of nodes
        Returns None if no valid edges or filtered out as noise.
    """
    avgWire = clusters["avgWire"].values.astype(np.float32)
    superlayer = clusters["superlayer"].values.astype(np.int32)
    track_ids_list = [parse_trkids(t) for t in clusters["trkIds"].values]
    num_clusters = len(clusters)

    # Normalize node features (0~1)
    wire_min, wire_max = avgWire.min(), avgWire.max()
    if wire_max - wire_min > 0:
        avgWire_norm = (avgWire - wire_min) / (wire_max - wire_min)
    else:
        return None

    # Node features: normalized avgWire + one-hot superlayer
    superlayer_onehot = np.eye(6)[superlayer - 1]
    x = np.concatenate([avgWire_norm.reshape(-1, 1), superlayer_onehot], axis=1)

    edges, edge_labels, edge_feats = [], [], []

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            superlayer_diff = superlayer[i] - superlayer[j]
            avgWire_diff = avgWire[i] - avgWire[j]

            if (abs(superlayer_diff) == 1 and abs(avgWire_diff) < avgWire_diff_max[0][superlayer[i]-1]) or \
               (abs(superlayer_diff) == 2 and abs(avgWire_diff) < avgWire_diff_max[1][superlayer[i]-1]):
                edges.append([i, j])
                label = 1 if (len(track_ids_list[i] & track_ids_list[j]) > 0) else 0
                edge_labels.append(label)
                edge_feats.append([superlayer_diff, avgWire_diff / (wire_max - wire_min)])

                if bidirectional:
                    edges.append([j, i])
                    edge_labels.append(label)
                    edge_feats.append([-superlayer_diff, -avgWire_diff / (wire_max - wire_min)])

    if len(edges) == 0:
        return None

    edges = np.array(edges)
    edge_labels = np.array(edge_labels)
    edge_feats = np.array(edge_feats, dtype=np.float32)

    # Keep some purely noisy events
    if edge_labels.sum() == 0 and random.random() > keep_all_noise_prob:
        return None

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edges.T, dtype=torch.long),
        edge_label=torch.tensor(edge_labels, dtype=torch.float),
        edge_attr=torch.tensor(edge_feats, dtype=torch.float),
        superlayer=torch.tensor(superlayer, dtype=torch.long),
        track_ids=[list(s) for s in track_ids_list]
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
    def __init__(self, in_channels=7, hidden_channels=32, num_layers=2, lr=1e-3, dropout=0.1):
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
    TorchScript 兼容的 EdgeClassifier Wrapper.

    接口:
        forward(x, edge_index, edge_attr=None) -> logits

    输入:
        x: Tensor [num_nodes, in_channels] 节点特征
        edge_index: LongTensor [2, num_edges] 边索引
        edge_attr: Optional Tensor [num_edges, edge_feat_dim] 边特征 (可选)

    输出:
        logits: Tensor [num_edges]
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.dropout = getattr(model, "dropout", 0.1)

        # 假设 model 有 convs (ModuleList), bns (ModuleList), classifier (Sequential)
        self.convs = model.convs
        self.bns = model.bns
        self.classifier = model.classifier

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor
    ) -> Tensor:
        # 如果 edge_attr 为空，则初始化 zeros
        if edge_attr.numel() == 0:
            edge_attr = torch.zeros((edge_index.size(1), 2), dtype=x.dtype, device=x.device)

        # conv + bn + relu + dropout
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        src, dst = edge_index
        edge_feat = torch.cat([x[src], x[dst], edge_attr], dim=1)
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




