from torch_geometric.data import Data, Batch

# -----------------------------
# Extract a single event from a batched Data object
# -----------------------------
def extract_event(batch: Batch, event_id: int) -> Data:
    """
    Extract a single event's data from a batched PyG Data object.

    Parameters
    ----------
    batch : torch_geometric.data.Batch
        Batched Data object containing multiple events.
    event_id : int
        Index of the event within the batch.

    Returns
    -------
    torch_geometric.data.Data
        Data object for the specified event, including:
        - x: node features
        - edge_index: edges for this event
        - edge_label: labels for edges (if present)
        - superlayer: node superlayer
        - track_ids: list of track ID sets
    """
    # Boolean mask of nodes belonging to this event
    node_mask = (batch.batch == event_id)

    # Mask edges where both endpoints are in this event
    edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]

    # Extract node features
    x = batch.x[node_mask]
    superlayer = batch.superlayer[node_mask]
    cluster_id = batch.cluster_id[node_mask]

    # Extract track_ids
    track_ids = []
    for tid, m in zip(batch.track_ids, node_mask):
        if m:
            if tid is None:
                track_ids.append([])
            elif isinstance(tid, list):
                track_ids.append(tid)
            else:  # 单个整数
                track_ids.append([tid])

    # Extract edges and edge labels
    edge_index = batch.edge_index[:, edge_mask]
    edge_label = batch.edge_label[edge_mask] if hasattr(batch, "edge_label") else None
    edge_attr = batch.edge_attr[edge_mask] if hasattr(batch, "edge_attr") else None

    # Build new Data object for this event
    event_data = Data(
        x=x,
        edge_index=edge_index,
        edge_label=edge_label,
        edge_attr=edge_attr,
        superlayer=superlayer,
        track_ids=track_ids,
        cluster_id=cluster_id
    )

    return event_data