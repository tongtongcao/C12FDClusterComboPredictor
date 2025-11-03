import torch
import networkx as nx
from typing import List, Tuple

class TrackPredictor:
    """
    Predict particle tracks from clusters (nodes) using a trained edge classifier,
    with per-edge probabilities and labels.

    Parameters
    ----------
    model : torch.nn.Module
        Trained edge classifier model.
    min_track_length : int
        Minimum number of nodes in a predicted track.
    threshold : float
        Edge probability threshold for considering an edge valid.
    max_overlap_nodes : int
        Maximum number of nodes allowed to overlap with already selected tracks.
    max_diff_nodes_close_tracks : int
        Maximum allowed differing nodes for similar tracks.
    prob_diff_threshold : float
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
    # Predict edge probabilities
    # -----------------------------
    def predict_edges(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = data.edge_index
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, torch.jit.ScriptModule):
                edge_attr = getattr(data, "edge_attr",
                                    torch.empty((edge_index.size(1), 3), dtype=data.x.dtype, device=data.x.device))
                logits = self.model(data.x, edge_index, edge_attr)
            else:
                logits = self.model(data)
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()
        return preds, probs

    # -----------------------------
    # Retain dense tracks while avoiding excessive overlap
    # -----------------------------
    def _retain_dense_tracks(self, all_tracks: List[List[int]],
                             edge_prob_map: dict,
                             edge_label_map: dict):
        final_tracks = []
        final_track_probs = []
        final_track_labels = []

        if not all_tracks:
            return final_tracks, final_track_probs, final_track_labels

        track_infos = []
        for tr in all_tracks:
            prob_list = [edge_prob_map.get((tr[i], tr[i + 1]), 0.0) for i in range(len(tr)-1)]
            label_list = [edge_label_map.get((tr[i], tr[i + 1]), 0) for i in range(len(tr)-1)]
            track_infos.append((len(tr), prob_list, label_list, tr))

        # Sort by length descending, then sum probability descending
        track_infos.sort(key=lambda x: (-x[0], -sum(x[1]) if x[1] else 0.0))

        node_usage_count = {}
        for length, prob_list, label_list, tr in track_infos:
            overlap_nodes = sum(node_usage_count.get(n, 0) for n in tr)
            if overlap_nodes <= self.max_overlap_nodes:
                final_tracks.append(tr)
                final_track_probs.append(prob_list)
                final_track_labels.append(label_list)
                for n in tr:
                    node_usage_count[n] = node_usage_count.get(n, 0) + 1
            else:
                can_keep = False
                for kept_tr, kept_probs, kept_labels in zip(final_tracks, final_track_probs, final_track_labels):
                    if len(tr) != len(kept_tr):
                        continue
                    diff_nodes = [n for n in tr if n not in kept_tr]
                    if len(diff_nodes) > self.max_diff_nodes_close_tracks:
                        continue
                    prob_diff_ok = all(abs(p1-p2) <= self.prob_diff_threshold for p1,p2 in zip(prob_list, kept_probs))
                    if prob_diff_ok:
                        can_keep = True
                        break
                if can_keep:
                    final_tracks.append(tr)
                    final_track_probs.append(prob_list)
                    final_track_labels.append(label_list)
                    for n in tr:
                        node_usage_count[n] = node_usage_count.get(n, 0) + 1

        return final_tracks, final_track_probs, final_track_labels

    # -----------------------------
    # Predict tracks (DFS + threshold)
    # -----------------------------
    def predict_tracks(self, data, threshold=None) -> Tuple[
        List[List[int]], List[List[float]], List[List[int]], List[int]]:
        if threshold is None:
            threshold = self.threshold

        preds, probs = self.predict_edges(data)
        edge_index = data.edge_index.t().tolist()
        num_nodes = data.x.size(0)
        superlayer = data.superlayer.cpu().numpy()

        # adjacency & edge maps
        adj = {i: set() for i in range(num_nodes)}
        edge_prob_map = {}
        if hasattr(data, "edge_label"):
            edge_label_map = {(i, j): int(lbl) for (i, j), lbl in zip(edge_index, data.edge_label.tolist())}
        else:
            edge_label_map = {(i, j): 0 for (i, j) in edge_index}

        for (i, j), p in zip(edge_index, probs.tolist()):
            if superlayer[j] > superlayer[i] and p >= threshold:
                adj[i].add(j)
            edge_prob_map[(i, j)] = p

        sl_to_nodes = {}
        for n in range(num_nodes):
            sl_to_nodes.setdefault(int(superlayer[n]), []).append(n)

        all_tracks = []

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
            if not extended and len(track) >= self.min_track_length:
                all_tracks.append(track.copy())

        for start_layer in [1, 2]:
            for start_node in sl_to_nodes.get(start_layer, []):
                dfs([start_node], used_sl={start_layer}, used_edges=set())

        final_tracks, final_track_probs, final_track_labels = self._retain_dense_tracks(
            all_tracks, edge_prob_map, edge_label_map
        )

        used_nodes_set = set(n for tr in final_tracks for n in tr)
        noise_hits = [n for n in range(num_nodes) if n not in used_nodes_set]

        if hasattr(data, "cluster_id"):
            final_tracks_cluster = [[data.cluster_id[n].item() for n in tr] for tr in final_tracks]
            noise_hits_cluster = [data.cluster_id[n].item() for n in noise_hits]
            return final_tracks_cluster, final_track_probs, final_track_labels, noise_hits_cluster
        else:
            return final_tracks, final_track_probs, final_track_labels, noise_hits

    # -----------------------------
    # Predict tracks using maximum spanning tree
    # -----------------------------
    def predict_tracks_with_max_weight(self, data, threshold=None) -> Tuple[
        List[List[int]], List[List[float]], List[List[int]], List[int]]:
        if threshold is None:
            threshold = self.threshold

        preds, probs = self.predict_edges(data)
        edge_index = data.edge_index.t().tolist()
        num_nodes = data.x.size(0)
        superlayer = data.superlayer.cpu().numpy()

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edge_prob_map = {}
        if hasattr(data, "edge_label"):
            edge_label_map = {(i, j): int(lbl) for (i, j), lbl in zip(edge_index, data.edge_label.tolist())}
        else:
            edge_label_map = {(i, j): 0 for (i, j) in edge_index}

        for (i, j), p in zip(edge_index, probs.tolist()):
            if superlayer[j] > superlayer[i] and p >= threshold:
                G.add_edge(i, j, weight=p)
            edge_prob_map[(i, j)] = p

        mst = nx.maximum_spanning_tree(G, weight='weight')

        adj = {i: set() for i in range(num_nodes)}
        for u, v in mst.edges():
            adj[u].add(v)
            adj[v].add(u)

        sl_to_nodes = {}
        for n in range(num_nodes):
            sl_to_nodes.setdefault(int(superlayer[n]), []).append(n)

        all_tracks = []

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
                    used_edges.add((next_node, last_node))
                    dfs(track, used_sl, used_edges)
                    track.pop()
                    used_sl.remove(next_sl)
                    used_edges.remove(edge)
                    used_edges.remove((next_node, last_node))
                    extended = True
            if not extended and len(track) >= self.min_track_length:
                all_tracks.append(track.copy())

        for start_layer in [1, 2]:
            for start_node in sl_to_nodes.get(start_layer, []):
                dfs([start_node], used_sl={start_layer}, used_edges=set())

        final_tracks, final_track_probs, final_track_labels = self._retain_dense_tracks(
            all_tracks, edge_prob_map, edge_label_map
        )

        used_nodes_set = set(n for tr in final_tracks for n in tr)
        noise_hits = [n for n in range(num_nodes) if n not in used_nodes_set]

        if hasattr(data, "cluster_id"):
            final_tracks_cluster = [[data.cluster_id[n].item() for n in tr] for tr in final_tracks]
            noise_hits_cluster = [data.cluster_id[n].item() for n in noise_hits]
            return final_tracks_cluster, final_track_probs, final_track_labels, noise_hits_cluster
        else:
            return final_tracks, final_track_probs, final_track_labels, noise_hits
