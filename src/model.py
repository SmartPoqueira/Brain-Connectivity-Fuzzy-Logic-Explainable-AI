"""
Brain connectivity classification models.

Implements Logistic Regression, SVM, MLP, Random Forest (matrix-based)
and GCN, GAT, GIN (graph-based) classifiers for preterm/term classification.
"""

import argparse
import json
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .fuzzy import fuzzy_labels, hard_labels
from .data_loader import load_dhcp_data, load_spatial_coordinates


# ──────────────────────────────────────────────────────────────
# Matrix-Based Models (sklearn)
# ──────────────────────────────────────────────────────────────

def extract_upper_triangle(matrices):
    """Extract upper-triangle features from (N, 90, 90) connectivity matrices."""
    n = matrices.shape[0]
    rows, cols = np.triu_indices(90, k=1)
    return matrices[:, rows, cols].reshape(n, -1)


def get_ml_model(name, use_fuzzy):
    """Return a sklearn model (regressor if fuzzy, classifier otherwise)."""
    if use_fuzzy:
        models = {
            "lr": Ridge(alpha=1.0, random_state=42),
            "svm": SVR(C=1.0, kernel="rbf"),
            "mlp": MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=1000, random_state=42),
            "rf": RandomForestRegressor(n_estimators=100, random_state=42),
        }
    else:
        models = {
            "lr": LogisticRegression(max_iter=5000, random_state=42),
            "svm": SVC(kernel="rbf", random_state=42, probability=True),
            "mlp": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, random_state=42),
            "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        }
    return models[name]


def run_ml_experiment(X, y, model_name, spatial_coords=None, use_fuzzy=False,
                      ga_values=None, n_splits=5, tau=37, T=1.0):
    """Run matrix-based ML experiment with stratified k-fold CV."""
    if spatial_coords is not None:
        spatial_flat = spatial_coords.ravel()
        X_spatial = np.tile(spatial_flat, (X.shape[0], 1))
        X_exp = np.hstack([X, X_spatial])
    else:
        X_exp = X

    if use_fuzzy and ga_values is not None:
        y_target = fuzzy_labels(ga_values, tau=tau, T=T)
    else:
        y_target = y

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for train_idx, test_idx in skf.split(X_exp, y):
        model = get_ml_model(model_name, use_fuzzy)
        model.fit(X_exp[train_idx], y_target[train_idx])
        
        if use_fuzzy:
            y_pred_cont = model.predict(X_exp[test_idx])
            y_pred = (y_pred_cont > 0.5).astype(int)
        else:
            y_pred = model.predict(X_exp[test_idx])
            
        acc = accuracy_score(y[test_idx], y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y[test_idx], y_pred, average=None, zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y[test_idx], y_pred, average="weighted", zero_division=0)
        
        results.append({
            "accuracy": acc, 
            "precision": p.tolist(), 
            "recall": r.tolist(), 
            "f1": f1.tolist(),
            "precision_weighted": p_w,
            "recall_weighted": r_w,
            "f1_weighted": f1_w
        })

    return results


# ──────────────────────────────────────────────────────────────
# Graph-Based Models (PyTorch Geometric)
# ──────────────────────────────────────────────────────────────

class BrainGAT(nn.Module):
    """Graph Attention Network for brain connectivity classification."""

    def __init__(self, num_node_features, hidden_dim=64, num_heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)


class BrainGCN(nn.Module):
    """Graph Convolutional Network for brain connectivity classification."""

    def __init__(self, num_node_features, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)


class BrainGIN(nn.Module):
    """Graph Isomorphism Network for brain connectivity classification."""

    def __init__(self, num_node_features, hidden_dim=64, dropout=0.3):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(num_node_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)


def connectivity_to_graph(matrix, node_features=None, threshold=0.0):
    """Convert a 90x90 connectivity matrix to a PyG Data object."""
    n = matrix.shape[0]
    edges_src, edges_dst, edge_weights = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] > threshold:
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
                edge_weights.extend([matrix[i, j], matrix[i, j]])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)

    if node_features is None:
        x = torch.eye(n, dtype=torch.float32)
    else:
        x = torch.tensor(node_features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def run_gnn_experiment(X, y, model_name, spatial_coords=None, use_fuzzy=False,
                       ga_values=None, n_splits=5, tau=37, T=1.0, epochs=30, lr=0.005):
    """Run graph-based GNN experiment with stratified k-fold CV."""
    if use_fuzzy and ga_values is not None:
        y_soft = fuzzy_labels(ga_values, tau=tau, T=T)
    else:
        y_soft = y.astype(float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_graphs = []
        for i in train_idx:
            data = connectivity_to_graph(X[i], node_features=spatial_coords)
            data.y = torch.tensor(y_soft[i], dtype=torch.float32)
            data.y_bin = torch.tensor(y[i], dtype=torch.float32)
            train_graphs.append(data)

        test_graphs = []
        for i in test_idx:
            data = connectivity_to_graph(X[i], node_features=spatial_coords)
            data.y = torch.tensor(y_soft[i], dtype=torch.float32)
            data.y_bin = torch.tensor(y[i], dtype=torch.float32)
            test_graphs.append(data)

        train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

        num_node_features = 3 if spatial_coords is not None else 90
        if model_name == "gat":
            model = BrainGAT(num_node_features=num_node_features)
        elif model_name == "gcn":
            model = BrainGCN(num_node_features=num_node_features)
        elif model_name == "gin":
            model = BrainGIN(num_node_features=num_node_features)
        else:
            raise ValueError(f"Unknown GNN model: {model_name}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()

        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                pred_prob = torch.sigmoid(out)
                pred_class = (pred_prob > 0.5).long().cpu().numpy()
                preds.extend(pred_class)
                true_labels.extend(batch.y_bin.long().cpu().numpy())

        preds = np.array(preds)
        true_labels = np.array(true_labels)

        acc = accuracy_score(true_labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(true_labels, preds, average=None, zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(true_labels, preds, average="weighted", zero_division=0)

        results.append({
            "accuracy": acc,
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f1.tolist(),
            "precision_weighted": p_w,
            "recall_weighted": r_w,
            "f1_weighted": f1_w
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Brain connectivity fuzzy label experiments")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seeds
    np.random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))

    # Load data
    mat_path = config["data"]["mat_path"]
    atlas_path = config["data"]["atlas_path"]
    matrices, ga, sex, pma = load_dhcp_data(mat_path)
    spatial_coords = load_spatial_coordinates(atlas_path) if config["spatial"]["use_coordinates"] else None

    # Get labels
    tau = config["labels"]["ga_threshold"]
    use_fuzzy = config["labels"]["use_fuzzy"]
    T = config["labels"].get("fuzzy_temperature", 1.0)
    y = hard_labels(ga, tau=tau)

    # Extract features for ML
    X_ml = extract_upper_triangle(matrices)
    n_splits = config["evaluation"]["n_splits"]

    # Results dict
    all_results = {}

    print(f"\n=======================================================")
    print(f"Running Experiments (use_fuzzy={use_fuzzy}, tau={tau}, T={T})")
    print(f"=======================================================")

    # Run ML experiments
    ml_models = config["models"].get("ml", [])
    for model_name in ml_models:
        print(f"\nTraining ML model: {model_name}...")
        # Without Spatial Coordinates
        res_baseline = run_ml_experiment(X_ml, y, model_name, spatial_coords=None,
                                         use_fuzzy=use_fuzzy, ga_values=ga, n_splits=n_splits, tau=tau, T=T)
        acc_b = np.mean([r["accuracy"] for r in res_baseline])
        f1_b = np.mean([r["f1_weighted"] for r in res_baseline])
        
        # With Spatial Coordinates
        res_spatial = run_ml_experiment(X_ml, y, model_name, spatial_coords=spatial_coords,
                                        use_fuzzy=use_fuzzy, ga_values=ga, n_splits=n_splits, tau=tau, T=T)
        acc_s = np.mean([r["accuracy"] for r in res_spatial])
        f1_s = np.mean([r["f1_weighted"] for r in res_spatial])

        print(f"  [Baseline] Accuracy: {acc_b:.4f} | Weighted F1: {f1_b:.4f}")
        print(f"  [Spatial ] Accuracy: {acc_s:.4f} | Weighted F1: {f1_s:.4f}")

        all_results[model_name] = {
            "baseline": res_baseline,
            "spatial": res_spatial
        }

    # Run GNN experiments
    gnn_models = config["models"].get("gnn", [])
    for model_name in gnn_models:
        print(f"\nTraining GNN model: {model_name}...")
        # Without Spatial Coordinates
        res_baseline = run_gnn_experiment(matrices, y, model_name, spatial_coords=None,
                                          use_fuzzy=use_fuzzy, ga_values=ga, n_splits=n_splits, tau=tau, T=T)
        acc_b = np.mean([r["accuracy"] for r in res_baseline])
        f1_b = np.mean([r["f1_weighted"] for r in res_baseline])
        
        # With Spatial Coordinates
        res_spatial = run_gnn_experiment(matrices, y, model_name, spatial_coords=spatial_coords,
                                         use_fuzzy=use_fuzzy, ga_values=ga, n_splits=n_splits, tau=tau, T=T)
        acc_s = np.mean([r["accuracy"] for r in res_spatial])
        f1_s = np.mean([r["f1_weighted"] for r in res_spatial])

        print(f"  [Baseline] Accuracy: {acc_b:.4f} | Weighted F1: {f1_b:.4f}")
        print(f"  [Spatial ] Accuracy: {acc_s:.4f} | Weighted F1: {f1_s:.4f}")

        all_results[model_name] = {
            "baseline": res_baseline,
            "spatial": res_spatial
        }

    # Output results
    output_dir = config.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved all results to {out_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
