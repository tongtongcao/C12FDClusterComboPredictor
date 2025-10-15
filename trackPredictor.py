import torch
import networkx as nx
import numpy as np

class TrackPredictor:
    """
    TrackPredictor uses a trained edge classifier to predict particle tracks
    from clusters (nodes) by generating candidate edges, predicting edge probabilities,
    and finding connected components or maximum spanning tree for track reconstruction.

    Parameters
    ----------
    model : nn.Module
        Trained EdgeClassifier model that predicts edge probabilities.
    min_track_length : int, default=5
        Minimum number of nodes in a predicted track (connected component).
    threshold : float, default=0.5
        Edge probability threshold for deciding if an edge exists.
    avgWire_diff_max : list[list[float]], default=[[12.0, 20.0, 12.0, 38.0, 14.0],[14.0, 18.0, 40.0, 40.0]]
        Maximum allowed avgWire difference for candidate edges.
        Format: [[for adjacent layers], [for skip layers]].
    bidirectional : bool, default=True
        Whether to generate edges in both directions (i→j and j→i).
    use_existing_edges : bool, default=True
        If True, uses `data.edge_index` if it exists instead of generating new candidate edges.
    """

    def __init__(self, model, min_track_length=5, threshold=0.5,
                 avgWire_diff_max=[[12.0, 20.0, 12.0, 38.0, 14.0],[14.0, 18.0, 40.0, 40.0]],
                 bidirectional=True, use_existing_edges=False):
        self.model = model
        self.avgWire_diff_max = avgWire_diff_max
        self.min_track_length = min_track_length
        self.threshold = threshold
        self.bidirectional = bidirectional
        self.use_existing_edges = use_existing_edges

    # -----------------------------
    # Generate candidate edges (if edge_index is not already provided)
    # -----------------------------
    def generate_candidate_edges(self, data) -> torch.Tensor:
        """
        Generate candidate edges based on superlayer adjacency and avgWire differences.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph with node features `x` and optional `superlayer`.

        Returns
        -------
        torch.Tensor
            Edge index tensor of shape [2, num_edges], possibly empty.
        """
        if self.use_existing_edges and hasattr(data, "edge_index") and data.edge_index.numel() > 0:
            return data.edge_index  # Use existing edges

        num_nodes = data.x.size(0)
        edges = []

        avgWire = data.x[:, 0].cpu().numpy()
        superlayer = data.superlayer.cpu().numpy()

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                superlayer_diff = superlayer[i] - superlayer[j]
                avgWire_diff = avgWire[i] - avgWire[j]

                if (abs(superlayer_diff) == 1 and abs(avgWire_diff) < self.avgWire_diff_max[0][superlayer[i] - 1]) or (
                        abs(superlayer_diff) == 2 and abs(avgWire_diff) < self.avgWire_diff_max[1][superlayer[i] - 1]):
                    edges.append([i, j])
                    if self.bidirectional:
                        edges.append([j, i])

        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    # -----------------------------
    # Predict edge probabilities
    # -----------------------------
    def predict_edges(self, data):
        edge_index = self.generate_candidate_edges(data)
        data.edge_index = edge_index
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, torch.jit.ScriptModule):
                # TorchScript wrapper expects x, edge_index, edge_attr
                edge_attr = getattr(data, "edge_attr",
                                    torch.empty((edge_index.size(1), 2), dtype=data.x.dtype, device=data.x.device))
                logits = self.model(data.x, edge_index, edge_attr)
            else:
                # Regular PyTorch model can accept Data or Batch
                logits = self.model(data)
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()
        return preds, probs

    def predict_tracks(self, data, threshold=None):
        """
        Predict tracks:
        - edges cannot be shared
        - if multiple tracks share nodes, keep the longest track
        """
        if threshold is None:
            threshold = self.threshold

        preds, probs = self.predict_edges(data)
        edge_index = data.edge_index.t().tolist()
        num_nodes = data.x.size(0)
        superlayer = data.superlayer.cpu().numpy()

        # 构建邻接表，只保留方向性和阈值
        edge_prob_map = {}  # (i,j) -> p
        adj = {i: set() for i in range(num_nodes)}
        for (i, j), p in zip(edge_index, probs):
            if p >= threshold and superlayer[j] > superlayer[i]:
                adj[i].add(j)
                edge_prob_map[(i, j)] = p

        # superlayer -> nodes
        sl_to_nodes = {}
        for n in range(num_nodes):
            sl_to_nodes.setdefault(int(superlayer[n]), []).append(n)

        all_tracks = []

        # DFS 生成所有轨迹
        def dfs(track, used_sl, used_edges):
            last_node = track[-1]
            extended = False
            for next_node in adj[last_node]:
                edge = (last_node, next_node)
                next_sl = int(superlayer[next_node])
                if next_sl not in used_sl and edge not in used_edges:
                    track.append(next_node)
                    used_sl.add(next_sl)
                    used_edges.add(edge)
                    dfs(track, used_sl, used_edges)
                    track.pop()
                    used_sl.remove(next_sl)
                    used_edges.remove(edge)
                    extended = True
            if len(track) >= self.min_track_length:
                all_tracks.append(track.copy())

        for start_node in sl_to_nodes.get(1, []):
            dfs([start_node], used_sl={1}, used_edges=set())

        # -----------------------------
        # 节点共享处理：保留最长轨迹
        # -----------------------------
        final_tracks = []
        used_nodes_set = set()

        # 按长度降序排序
        sorted_tracks = sorted(all_tracks, key=lambda tr: -len(tr))

        for tr in sorted_tracks:
            shared_nodes = sum(1 for n in tr if n in used_nodes_set)
            if shared_nodes == 0:
                final_tracks.append(tr)
                used_nodes_set.update(tr)
            else:
                # 节点共享，保留最长轨迹（已按长度排序，所以跳过即可）
                continue

        # 剩余节点为噪声
        noise_hits = [n for n in range(num_nodes) if n not in used_nodes_set]

        # -----------------------------
        # 将节点索引映射为 cluster_id
        # -----------------------------
        if hasattr(data, "cluster_id"):
            final_tracks_cluster = [[data.cluster_id[n].item() for n in tr] for tr in final_tracks]
            noise_hits_cluster = [data.cluster_id[n].item() for n in noise_hits]
            return final_tracks_cluster, noise_hits_cluster

        else:
            # 如果没有 cluster_id，就返回原来的节点索引
            return final_tracks, noise_hits

    def predict_tracks_with_max_weight(self, data, threshold=None) -> tuple[list[list[int]], list[int]]:
        """
        Predict tracks using maximum spanning tree weighted by edge probabilities,
        but applying rules:
        - edges cannot be shared
        - nodes can be shared, but keep only longest track in case of shared nodes
        Returns tracks and noise as cluster_id.
        """
        if threshold is None:
            threshold = self.threshold

        # ---------- 预测边 ----------
        preds, probs = self.predict_edges(data)
        edge_index = data.edge_index.t().tolist()
        num_nodes = data.x.size(0)
        superlayer = data.superlayer.cpu().numpy()

        # ---------- 构建加权图 ----------
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for (i, j), p in zip(edge_index, probs.tolist()):
            if p >= threshold and superlayer[j] > superlayer[i]:
                if self.bidirectional and i > j:
                    continue
                G.add_edge(i, j, weight=p)

        # ---------- 最大生成树 ----------
        mst = nx.maximum_spanning_tree(G, weight='weight')

        # ---------- 邻接表 ----------
        adj = {i: set() for i in range(num_nodes)}
        edge_weight_map = {}
        for u, v, d in mst.edges(data=True):
            adj[u].add(v)
            adj[v].add(u)  # 保留无向邻居
            edge_weight_map[(u, v)] = d['weight']
            edge_weight_map[(v, u)] = d['weight']

        # ---------- DFS 搜索轨迹 ----------
        all_tracks = []
        sl_to_nodes = {}
        for n in range(num_nodes):
            sl_to_nodes.setdefault(int(superlayer[n]), []).append(n)

        def dfs(track, used_sl, used_edges):
            last_node = track[-1]
            for next_node in adj[last_node]:
                edge = (last_node, next_node)
                next_sl = int(superlayer[next_node])
                if next_sl not in used_sl and edge not in used_edges:
                    track.append(next_node)
                    used_sl.add(next_sl)
                    used_edges.add(edge)
                    used_edges.add((next_node, last_node))
                    dfs(track, used_sl, used_edges)
                    track.pop()
                    used_sl.remove(next_sl)
                    used_edges.remove(edge)
                    used_edges.remove((next_node, last_node))
            if len(track) >= self.min_track_length:
                all_tracks.append(track.copy())

        for start_node in sl_to_nodes.get(1, []):
            dfs([start_node], used_sl={1}, used_edges=set())

        # ---------- 节点共享处理：保留最长轨迹 ----------
        final_tracks = []
        used_nodes_set = set()
        sorted_tracks = sorted(all_tracks, key=lambda tr: -len(tr))

        for tr in sorted_tracks:
            shared_nodes = sum(1 for n in tr if n in used_nodes_set)
            if shared_nodes == 0:
                final_tracks.append(tr)
                used_nodes_set.update(tr)

        # ---------- 剩余节点为噪声 ----------
        noise_hits = [n for n in range(num_nodes) if n not in used_nodes_set]

        # ---------- 映射为 cluster_id ----------
        if hasattr(data, "cluster_id"):
            final_tracks_cluster = [[data.cluster_id[n].item() for n in tr] for tr in final_tracks]
            noise_hits_cluster = [data.cluster_id[n].item() for n in noise_hits]
            return final_tracks_cluster, noise_hits_cluster
        else:
            return final_tracks, noise_hits

