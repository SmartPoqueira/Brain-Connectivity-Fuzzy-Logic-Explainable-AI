"""
Brain connectivity classification models.

Implements Logistic Regression, SVM, MLP, Random Forest (matrix-based)
and GCN, GAT, GIN (graph-based) classifiers for preterm/term classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .fuzzy import fuzzy_labels


# ──────────────────────────────────────────────────────────────
# Matrix-Based Models (sklearn)
# ──────────────────────────────────────────────────────────────

def extract_upper_triangle(matrices):
    """Extract upper-triangle features from (N, 90, 90) connectivity matrices."""
    n = matrices.shape[0]
    rows, cols = np.triu_indices(90, k=1)
    return matrices[:, rows, cols].reshape(n, -1)


def get_ml_classifier(name):
    """Return a sklearn classifier by name."""
    classifiers = {
        "lr": LogisticRegression(max_iter=5000, random_state=42),
        "svm": SVC(kernel="rbf", random_state=42, probability=True),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    return classifiers[name]


def run_ml_experiment(X, y, model_name, spatial_coords=None, use_fuzzy=False,
                      ga_values=None, n_splits=5, tau=37, T=1.0):
    """Run matrix-based ML experiment with stratified k-fold CV."""
    if spatial_coords is not None:
        X = np.hstack([X, np.tile(spatial_coords, (X.shape[0], 1))])

    if use_fuzzy and ga_values is not None:
        y_soft = fuzzy_labels(ga_values, tau=tau, T=T)
        y_train_labels = (y_soft > 0.5).astype(int)
    else:
        y_train_labels = y

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for train_idx, test_idx in skf.split(X, y):
        clf = get_ml_classifier(model_name)
        clf.fit(X[train_idx], y_train_labels[train_idx])
        y_pred = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y[test_idx], y_pred, average=None)
        results.append({"accuracy": acc, "precision": p, "recall": r, "f1": f1})

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
