import torch
import networkx as nx
from typing import List, Tuple

class TrackPredictor:
    """
    Predict particle tracks from clusters (nodes) using a trained edge classifier,
    existing edges, and edge probabilities, with support for dense track retention
    and bidirectional edge scoring.

    Parameters
    ----------
    model : torch.nn.Module
        Trained edge classifier model.
    min_track_length : int, default=5
        Minimum number of nodes in a predicted track.
    threshold : float, default=0.5
        Edge probability threshold for considering an edge valid.
    max_overlap_nodes : int, default=1
        Maximum number of nodes allowed to overlap with already selected tracks.
    max_diff_nodes_close_tracks : int, default=1
        Maximum number of differing nodes for tracks similar to already selected tracks.
    prob_diff_threshold : float, default=0.1
        Maximum allowed probability difference per edge for similar tracks.
    """

    def __init__(self, model, min_track_length=5, threshold=0.15,
                 max_overlap_nodes=1, max_diff_nodes_close_tracks=1,
                 prob_diff_threshold=0.08):
        self.model = model
        self.min_track_length = min_track_length
        self.threshold = threshold
        self.max_overlap_nodes = max_overlap_nodes
        self.max_diff_nodes_close_tracks = max_diff_nodes_close_tracks
        self.prob_diff_threshold = prob_diff_threshold

    # -----------------------------
    # Bidirectional edge scoring
    # -----------------------------
    def compute_bidirectional_edge_score(self, edge: Tuple[int, int], edge_prob_map: dict, method='sum') -> float:
        p1 = edge_prob_map.get(edge, 0.0)
        p2 = edge_prob_map.get((edge[1], edge[0]), 0.0)
        if method == 'sum':
            return p1 + p2
        elif method == 'mean':
            return (p1 + p2) / 2
        else:
            raise ValueError(f"Unknown method {method}")

    # -----------------------------
    # Predict edge probabilities
    # -----------------------------
    def predict_edges(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = data.edge_index
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, torch.jit.ScriptModule):
                edge_attr = getattr(data, "edge_attr",
                                    torch.empty((edge_index.size(1), 2), dtype=data.x.dtype, device=data.x.device))
                logits = self.model(data.x, edge_index, edge_attr)
            else:
                logits = self.model(data)
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()
        return preds, probs

    # -----------------------------
    # Dense track retention (新版本)
    # -----------------------------
    def _retain_dense_tracks(self, all_tracks: List[List[int]], edge_prob_map: dict):
        """
        Select a subset of tracks to avoid excessive overlap while retaining dense tracks.
        Similar tracks are allowed if length is same, overlapping nodes <= max_diff_nodes_close_tracks,
        and per-edge probabilities difference <= prob_diff_threshold.

        Parameters
        ----------
        all_tracks : list of list of int
            Candidate tracks (node indices).
        edge_prob_map : dict
            Dictionary mapping edge (i,j) to probability.

        Returns
        -------
        final_tracks : list of list of int
            Selected tracks.
        final_track_probs : list of list of float
            Per-edge probabilities for each track.
        """
        final_tracks = []
        final_track_probs = []

        if not all_tracks:
            return final_tracks, final_track_probs

        track_infos = []
        for tr in all_tracks:
            prob_list = [edge_prob_map.get((tr[i], tr[i + 1]), 0.0) for i in range(len(tr) - 1)]
            track_infos.append((len(tr), prob_list, tr))

        # Sort by length descending, then max edge probability descending
        track_infos.sort(key=lambda x: (-x[0], -max(x[1]) if x[1] else 0.0))

        node_usage_count = {}

        for length, prob_list, tr in track_infos:
            overlap_nodes = sum(node_usage_count.get(n, 0) for n in tr)
            if overlap_nodes <= self.max_overlap_nodes:
                final_tracks.append(tr)
                final_track_probs.append(prob_list)
                for n in tr:
                    node_usage_count[n] = node_usage_count.get(n, 0) + 1
            else:
                can_keep = False
                for kept_tr, kept_probs in zip(final_tracks, final_track_probs):
                    if len(tr) != len(kept_tr):
                        continue
                    diff_nodes = [n for n in tr if n not in kept_tr]
                    if len(diff_nodes) > self.max_diff_nodes_close_tracks:
                        continue
                    prob_diff_ok = all(
                        abs(p1 - p2) <= self.prob_diff_threshold for p1, p2 in zip(prob_list, kept_probs))
                    if prob_diff_ok:
                        can_keep = True
                        break
                if can_keep:
                    final_tracks.append(tr)
                    final_track_probs.append(prob_list)
                    for n in tr:
                        node_usage_count[n] = node_usage_count.get(n, 0) + 1

        return final_tracks, final_track_probs

    def predict_tracks(self, data, threshold=None) -> Tuple[List[List[int]], List[List[float]], List[int]]:
        """
        Predict tracks using directed edges (smaller layer -> larger layer),
        apply dense track retention, and return per-edge probabilities.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data containing `x`, `edge_index`, `superlayer`, and optionally `cluster_id`.
        threshold : float, optional
            Edge probability threshold.

        Returns
        -------
        final_tracks_cluster : list of list of int
            Selected tracks with node indices replaced by `cluster_id` if available.
        final_tracks_probs : list of list of float
            Per-edge probabilities for each track.
            Each inner list has length len(track)-1.
        noise_hits_cluster : list of int
            Nodes not assigned to any track.
        """
        if threshold is None:
            threshold = self.threshold

        preds, probs = self.predict_edges(data)
        edge_index = data.edge_index.t().tolist()
        num_nodes = data.x.size(0)
        superlayer = data.superlayer.cpu().numpy()

        edge_prob_map = {}
        adj = {i: set() for i in range(num_nodes)}
        for (i, j), p in zip(edge_index, probs):
            if p >= threshold and superlayer[j] > superlayer[i]:
                adj[i].add(j)
                edge_prob_map[(i, j)] = p

        sl_to_nodes = {}
        for n in range(num_nodes):
            sl_to_nodes.setdefault(int(superlayer[n]), []).append(n)

        all_tracks = []
        all_track_probs = []

        def dfs(track, track_probs, used_sl, used_edges):
            last_node = track[-1]
            extended = False
            for next_node in adj[last_node]:
                edge = (last_node, next_node)
                next_sl = int(superlayer[next_node])
                if next_sl not in used_sl and edge not in used_edges:
                    track.append(next_node)
                    track_probs.append(edge_prob_map[edge])
                    used_sl.add(next_sl)
                    used_edges.add(edge)
                    dfs(track, track_probs, used_sl, used_edges)
                    track.pop()
                    track_probs.pop()
                    used_sl.remove(next_sl)
                    used_edges.remove(edge)
                    extended = True
            if not extended and len(track) >= self.min_track_length:
                all_tracks.append(track.copy())
                all_track_probs.append(track_probs.copy())

        # 从第1层和第2层节点出发
        for start_layer in [1, 2]:
            for start_node in sl_to_nodes.get(start_layer, []):
                dfs([start_node], [], used_sl={start_layer}, used_edges=set())

        final_tracks, final_track_probs = self._retain_dense_tracks(all_tracks, edge_prob_map)

        used_nodes_set = set(n for tr in final_tracks for n in tr)
        noise_hits = [n for n in range(num_nodes) if n not in used_nodes_set]

        if hasattr(data, "cluster_id"):
            final_tracks_cluster = [[data.cluster_id[n].item() for n in tr] for tr in final_tracks]
            noise_hits_cluster = [data.cluster_id[n].item() for n in noise_hits]
            return final_tracks_cluster, final_track_probs, noise_hits_cluster
        else:
            return final_tracks, final_track_probs, noise_hits

    # -----------------------------
    # Predict tracks using MST
    # -----------------------------
    def predict_tracks_with_max_weight(self, data, threshold=None) -> Tuple[
        List[List[int]], List[List[float]], List[int]]:
        """
        Predict tracks using maximum spanning tree weighted by edge probabilities,
        apply dense track retention, and return per-edge probabilities.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data containing `x`, `edge_index`, `superlayer`, and optionally `cluster_id`.
        threshold : float, optional
            Edge probability threshold.

        Returns
        -------
        final_tracks_cluster : list of list of int
            Selected tracks after MST and dense track retention.
        final_tracks_probs : list of list of float
            Per-edge probabilities for each track.
            Each inner list has length len(track)-1.
        noise_hits_cluster : list of int
            Nodes not assigned to any track.
        """
        if threshold is None:
            threshold = self.threshold

        preds, probs = self.predict_edges(data)
        edge_index = data.edge_index.t().tolist()
        num_nodes = data.x.size(0)
        superlayer = data.superlayer.cpu().numpy()

        # Build graph with edges above threshold
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edge_prob_map = {}
        for (i, j), p in zip(edge_index, probs.tolist()):
            if p >= threshold and superlayer[j] > superlayer[i]:
                G.add_edge(i, j, weight=p)
                edge_prob_map[(i, j)] = p
                edge_prob_map[(j, i)] = p

        mst = nx.maximum_spanning_tree(G, weight='weight')

        adj = {i: set() for i in range(num_nodes)}
        for u, v in mst.edges():
            adj[u].add(v)
            adj[v].add(u)

        sl_to_nodes = {}
        for n in range(num_nodes):
            sl_to_nodes.setdefault(int(superlayer[n]), []).append(n)

        all_tracks = []
        all_track_probs = []

        def dfs(track, track_probs, used_sl, used_edges):
            last_node = track[-1]
            extended = False
            for next_node in adj[last_node]:
                edge = (last_node, next_node)
                next_sl = int(superlayer[next_node])
                if next_sl not in used_sl and edge not in used_edges:
                    track.append(next_node)
                    track_probs.append(edge_prob_map[edge])
                    used_sl.add(next_sl)
                    used_edges.add(edge)
                    used_edges.add((next_node, last_node))
                    dfs(track, track_probs, used_sl, used_edges)
                    track.pop()
                    track_probs.pop()
                    used_sl.remove(next_sl)
                    used_edges.remove(edge)
                    used_edges.remove((next_node, last_node))
                    extended = True
            if not extended and len(track) >= self.min_track_length:
                all_tracks.append(track.copy())
                all_track_probs.append(track_probs.copy())

        # 从第1层和第2层出发
        for start_layer in [1, 2]:
            for start_node in sl_to_nodes.get(start_layer, []):
                dfs([start_node], [], used_sl={start_layer}, used_edges=set())

        final_tracks, final_track_probs = self._retain_dense_tracks(all_tracks, edge_prob_map)

        used_nodes_set = set(n for tr in final_tracks for n in tr)
        noise_hits = [n for n in range(num_nodes) if n not in used_nodes_set]

        if hasattr(data, "cluster_id"):
            final_tracks_cluster = [[data.cluster_id[n].item() for n in tr] for tr in final_tracks]
            noise_hits_cluster = [data.cluster_id[n].item() for n in noise_hits]
            return final_tracks_cluster, final_track_probs, noise_hits_cluster
        else:
            return final_tracks, final_track_probs, noise_hits

