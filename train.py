import os
import time
import random
import argparse
from collections import Counter
from typing import List, Tuple, Optional

import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import pandas as pd
import networkx as nx
from torch_geometric.data import Data, Batch

# -----------------------------
# Imports from your modules
# -----------------------------
from trainer import EdgeClassifier, EdgeClassifierWrapper, LossTracker, build_graph_from_hits
from plotter import Plotter
from trackPredictor import TrackPredictor
from event import extract_event


# -----------------------------
# Command-line argument parsing
# -----------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for GNN training and inference.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments including device, input files, hyperparameters,
        output directory, and plotting options.
    """
    parser = argparse.ArgumentParser(description="GNN Training and Inference")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("inputs", type=str, nargs="*", default=["clusters_sector1.csv"])
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--end_name", type=str, default="")
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--keep_all_noise_prob", type=float, default=0)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--enable_progress_bar", action="store_true")
    parser.add_argument("--max_plot_events", type=int, default=25,
                        help="maximum number of validation events to plot (-1 for all)")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser.parse_args()


# -----------------------------
# Load CSV files and convert to PyG Data graphs
# -----------------------------
def load_graphs_from_csv(files: List[str], val_ratio: float = 0.2, keep_all_noise_prob: float = 0.1) -> Tuple[List[Data], List[Data]]:
    """
    Load multiple CSV files and convert events into PyG Data objects.

    Parameters
    ----------
    files : list[str]
        List of CSV filenames.
    val_ratio : float
        Fraction of events for validation.
    keep_all_noise_prob : float
        Probability to keep events with no positive edges.

    Returns
    -------
    tuple[list[Data], list[Data]]
        (train_graphs, val_graphs)
    """
    all_graphs: List[Data] = []
    for f in files:
        df = pd.read_csv(f)
        for _, hits in df.groupby("eventIdx"):
            data = build_graph_from_hits(hits, keep_all_noise_prob=keep_all_noise_prob)
            if data is not None:
                all_graphs.append(data)
    random.shuffle(all_graphs)
    num_val = int(len(all_graphs) * val_ratio)
    return all_graphs[num_val:], all_graphs[:num_val]


# -----------------------------
# Main function
# -----------------------------
def main() -> None:
    """
    Main script for training a GNN edge classifier and performing track prediction.

    Steps:
    1. Parse arguments and load CSV files into PyG Data graphs.
    2. Initialize model, loss tracker, and plotting utilities.
    3. Train the EdgeClassifier model (unless --no_train is set).
    4. Save the trained model to TorchScript.
    5. Evaluate validation set to compute best threshold using Precision/Recall/F1.
    6. Use TrackPredictor to predict tracks for validation events.
    7. Plot tracks for selected events and count tracks of length 5 and 6.
    """
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading data...")
    start_time = time.time()
    train_graphs, val_graphs = load_graphs_from_csv(
        args.inputs, val_ratio=args.val_ratio, keep_all_noise_prob=args.keep_all_noise_prob
    )
    print(f"Train size: {len(train_graphs)}, Val size: {len(val_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_graphs, batch_size=1, num_workers=0, persistent_workers=False)
    end_time = time.time()
    print(f"Data loading took {end_time - start_time:.2f}s\n")

    # -----------------------------
    # Initialize model
    # -----------------------------
    model = EdgeClassifier(
        in_channels=7,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        lr=args.lr,
        dropout=args.dropout
    )
    loss_tracker = LossTracker()
    plotter = Plotter(print_dir=args.outdir, end_name=args.end_name)

    # -----------------------------
    # Train model
    # -----------------------------
    ts_model_path = os.path.join(args.outdir, f"gnn_{args.end_name}.pt")
    if not args.no_train:
        # Device selection
        if args.device == "cpu":
            accelerator, devices = "cpu", 1
        elif args.device == "gpu":
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
            devices = 1
        else:
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
            devices = "auto" if torch.cuda.is_available() else 1

        print(f"Using accelerator={accelerator}, devices={devices}")

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=args.max_epochs,
            enable_progress_bar=args.enable_progress_bar,
            enable_checkpointing=False,
            logger=False,
            callbacks=[loss_tracker]
        )

        print("Starting training...")
        t0 = time.time()
        trainer.fit(model, train_loader, val_loader)
        t1 = time.time()
        print(f"Training finished in {(t1 - t0)/60:.2f} minutes")

        plotter.plotTrainLoss(loss_tracker)

        # TorchScript save
        model.to("cpu")
        wrapper = EdgeClassifierWrapper(model)
        wrapper.eval()

        try:
            torchscript_model = torch.jit.script(wrapper)
        except Exception as e:
            print("TorchScript scripting failed, trying trace...")
            # example for trace
            example_x = torch.randn(5, 7)  # node features
            example_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
            example_edge_attr = torch.randn(3, 2)
            torchscript_model = torch.jit.trace(wrapper, (example_x, example_edge_index, example_edge_attr))

        torchscript_model.save(ts_model_path)
        print(f"TorchScript model saved to {ts_model_path}")
    else:
        if not os.path.exists(ts_model_path):
            ts_model_path = os.path.join("nets", "gnn_default.pt")

    # -----------------------------
    # Load trained model for inference
    # -----------------------------
    print("Loading trained model...")
    model_ts = torch.jit.load(ts_model_path)
    model_ts.eval()

    # -----------------------------
    # Compute best threshold using validation set
    # -----------------------------
    print("\nComputing best threshold using validation set...")
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            edge_attr = getattr(batch, "edge_attr",
                                torch.empty((batch.edge_index.size(1), 2), dtype=batch.x.dtype, device=batch.x.device))
            out = model_ts(batch.x, batch.edge_index, edge_attr)
            out_prob = torch.sigmoid(out)
            y_true_all.append(batch.edge_label.cpu())
            y_pred_all.append(out_prob.cpu())

    y_true_all = torch.cat(y_true_all)
    y_pred_all = torch.cat(y_pred_all)

    plotter.plot_edge_probs(y_true_all, y_pred_all)

    precision, recall, f1, best_th = plotter.precision_recall_f1_with_best_threshold(y_true_all, y_pred_all)

    tpr_list, tnr_list, thresholds = plotter.tpr_tnr_vs_threshold(y_true_all, y_pred_all)

    # -----------------------------
    # TrackPredictor with best threshold
    # -----------------------------
    predictor = TrackPredictor(
        model=model_ts,
        min_track_length=5
    )

    plotter.track_purity_efficiency_vs_threshold(predictor, val_loader)

    predictor.threshold = 0.15

    all_tracks, all_noise = [], []
    length_counter = Counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for event_id in batch.batch.unique():
                event_data = extract_event(batch, event_id)
                tracks, track_probs, track_labels, noise_hits = predictor.predict_tracks(event_data)
                all_tracks.append(tracks)
                all_noise.append(noise_hits)

                # Track length statistics
                for tr in tracks:
                    length_counter[len(tr)] += 1

                if args.max_plot_events < 0 or batch_idx < args.max_plot_events:
                    save_path_edges = f"{plotter.print_dir}/edges_batch{batch_idx}_event{event_id.item()}_{plotter.end_name}.png"
                    save_path_tracks = f"{plotter.print_dir}/tracks_batch{batch_idx}_event{event_id.item()}_{plotter.end_name}.png"
                    plotter.plot_all_edges(predictor,
                                                  event_data,
                                                  title=f"Event {batch_idx}",
                                                  save_path=save_path_edges
                    )
                    plotter.plot_predicted_tracks(predictor,
                        event_data,
                        title=f"Event {batch_idx}",
                        save_path=save_path_tracks
                    )

    # Print number of tracks with length 5 and 6
    len_5 = length_counter.get(5, 0)
    len_6 = length_counter.get(6, 0)
    print(f"Number of tracks with length 5: {len_5}")
    print(f"Number of tracks with length 6: {len_6}")


if __name__ == "__main__":
    main()
