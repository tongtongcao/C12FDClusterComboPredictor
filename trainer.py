import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import random
from pytorch_lightning.callbacks import Callback


# -----------------------------
# Parse track IDs
# -----------------------------
def parse_trkids(trk_series: pd.Series) -> List[set[int]]:
    """
    Parse track IDs from a pandas Series of strings.

    Parameters
    ----------
    trk_series : pd.Series
        A pandas Series containing track ID strings (e.g., "1;2;3").

    Returns
    -------
    List[set[int]]
        A list of sets of integers representing parsed track IDs. Empty sets for NaN or invalid entries.
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
# Build graph from hits
# -----------------------------
def build_graph_from_hits(
    clusters: pd.DataFrame,
    keep_all_noise_prob: float = 0.1,
    avgWire_diff_max: Optional[List[List[float]]] = None,
    bidirectional: bool = True
) -> Optional[Data]:
    """
    Construct a PyTorch Geometric graph from cluster hits.

    Parameters
    ----------
    clusters : pd.DataFrame
        DataFrame containing cluster information with columns 'avgWire', 'slope', 'superlayer', 'clusterIdx', 'trkIds'.
    keep_all_noise_prob : float, default=0.1
        Probability to keep a graph with only negative edges (all noise).
    avgWire_diff_max : list[list[float]], optional
        Maximum allowed average wire differences for edges between superlayers.
    bidirectional : bool, default=True
        Whether to create bidirectional edges.

    Returns
    -------
    Optional[Data]
        A PyTorch Geometric Data object with node features, edge indices, edge features, and edge labels.
        Returns None if no valid edges exist.
    """
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
    slope = clusters["slope"].values.astype(np.float32)
    superlayer = clusters["superlayer"].values.astype(np.int32)
    cluster_ids = clusters["clusterIdx"].values.astype(np.int32)
    track_ids_list = parse_trkids(clusters["trkIds"])

    # Node features
    wire_range = 112.0
    superlayer_range = 6.0
    avgWire_norm = avgWire / wire_range
    superlayer_norm = superlayer / superlayer_range
    x = torch.tensor(np.stack([avgWire_norm, slope, superlayer_norm], axis=1), dtype=torch.float)

    # Edge construction
    L1 = superlayer[:, None]
    L2 = superlayer[None, :]
    dL = np.abs(L2 - L1)

    W1 = avgWire[:, None]
    W2 = avgWire[None, :]
    diff = W2 - W1

    mask = np.zeros((num_clusters, num_clusters), dtype=bool)
    for delta in range(1, len(avgWire_diff_max) + 1):
        idx = np.where(dL == delta)
        for i, j in zip(*idx):
            lower = min(superlayer[i], superlayer[j])
            if lower - 1 >= len(avgWire_diff_max[delta - 1]):
                continue
            max_diff = avgWire_diff_max[delta - 1][lower - 1]
            if abs(diff[i, j]) <= max_diff:
                mask[i, j] = True

    src, dst = np.where(np.triu(mask, k=1))

    if len(src) == 0:
        return None

    # Edge features
    sl_diff = superlayer[src] - superlayer[dst]
    aw_diff = avgWire[src] - avgWire[dst]
    slope_diff = slope[src] - slope[dst]
    edge_attr = np.stack([aw_diff / wire_range, slope_diff, sl_diff / superlayer_range], axis=1).astype(np.float32)

    # Edge labels
    edge_label = np.array([1 if len(track_ids_list[i] & track_ids_list[j]) > 0 else 0
                           for i, j in zip(src, dst)], dtype=np.float32)

    if edge_label.sum() == 0 and random.random() > keep_all_noise_prob:
        return None

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
# Edge classifier model
# -----------------------------
class EdgeClassifier(pl.LightningModule):
    """
    GraphSAGE-based edge classifier.
    """

    def __init__(self, in_channels=3, hidden_channels=32, num_layers=2, lr=1e-3, dropout=0.1):
        """
        Parameters
        ----------
        in_channels : int
            Number of input node features.
        hidden_channels : int
            Number of hidden channels in GNN layers.
        num_layers : int
            Number of GraphSAGE layers.
        lr : float
            Learning rate for optimizer.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        self.save_hyperparameters()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_ch, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels + 3, hidden_channels),
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
            A batch of graphs (PyTorch Geometric Batch object).

        Returns
        -------
        torch.Tensor
            Predicted logits for edges.
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

        edge_feat = torch.cat([x[src], x[dst], edge_attr.to(x.dtype)], dim=1)
        logits = self.classifier(edge_feat).view(-1)
        return logits

    def training_step(self, batch: Batch) -> torch.Tensor:
        logits = self(batch)
        labels = batch.edge_label.float()
        pos_count = (labels == 1.0).sum().item()
        neg_count = (labels == 0.0).sum().item()
        pos_weight = torch.tensor(max(neg_count / max(1.0, pos_count), 1.0),
                                  dtype=torch.float, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch: Batch) -> torch.Tensor:
        logits = self(batch)
        labels = batch.edge_label.float()
        pos_count = (labels == 1.0).sum().item()
        neg_count = (labels == 0.0).sum().item()
        pos_weight = torch.tensor(max(neg_count / max(1.0, pos_count), 1.0),
                                  dtype=torch.float, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=labels.size(0))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# -----------------------------
# Inference wrapper
# -----------------------------
class EdgeClassifierWrapper(nn.Module):
    """
    TorchScript-compatible EdgeClassifier wrapper.

    Forward signature:
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_feat_dim] or empty tensor

    Returns
    -------
    logits : [num_edges]
        Predicted logits for each edge.
    """

    def __init__(self, model: nn.Module, edge_feat_dim: int = 3):
        """
        Parameters
        ----------
        model : nn.Module
            Pretrained EdgeClassifier.
        edge_feat_dim : int
            Number of edge feature dimensions.
        """
        super().__init__()
        self.convs = model.convs
        self.bns = model.bns
        self.classifier = model.classifier
        self.dropout = getattr(model, "dropout", 0.1)
        self.edge_feat_dim = edge_feat_dim

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_attr.numel() == 0:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_feat_dim),
                                    dtype=x.dtype, device=x.device)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        src = edge_index[0]
        dst = edge_index[1]
        edge_feat = torch.cat([x[src], x[dst], edge_attr], dim=1)
        logits = self.classifier(edge_feat).view(-1)
        return logits


# -----------------------------
# Callback for tracking losses
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
