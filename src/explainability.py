"""
SHAP-based explainability pipeline for brain connectivity models.

Computes edge-importance matrices and node-level aggregation scores
to identify key brain regions driving preterm/term classification.
"""

import numpy as np
import shap


def compute_shap_values(model, X, feature_names=None):
    """Compute SHAP values for a fitted sklearn model.

    Parameters
    ----------
    model : fitted sklearn classifier
    X : np.ndarray of shape (n_samples, n_features)
    feature_names : list of str, optional

    Returns
    -------
    shap_values : np.ndarray of shape (n_samples, n_features)
    """
    explainer = shap.Explainer(model, X)
    sv = explainer(X)
    return sv.values


def edge_importance_matrix(shap_values, n_regions=90):
    """Compute mean absolute SHAP value for each edge.

    Parameters
    ----------
    shap_values : np.ndarray of shape (N, d)
        SHAP values for each sample and edge feature.
    n_regions : int
        Number of brain regions (nodes).

    Returns
    -------
    M : np.ndarray of shape (n_regions, n_regions)
        Symmetric edge importance matrix.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    M = np.zeros((n_regions, n_regions))
    rows, cols = np.triu_indices(n_regions, k=1)
    n_edges = len(rows)

    # Map SHAP values to upper triangle (edge features only)
    edge_shap = mean_abs[:n_edges]
    M[rows, cols] = edge_shap
    M[cols, rows] = edge_shap  # Symmetrize

    return M


def node_importance(M):
    """Aggregate edge importance to node level.

    For each node v: I_v = (1/(n-1)) * sum_{u != v} M_{uv}

    Parameters
    ----------
    M : np.ndarray of shape (n, n)
        Edge importance matrix.

    Returns
    -------
    I : np.ndarray of shape (n,)
        Node importance scores.
    """
    n = M.shape[0]
    I = M.sum(axis=1) / (n - 1)
    return I


def top_bottom_nodes(I, node_names, k=10):
    """Return top-k and bottom-k nodes by importance.

    Parameters
    ----------
    I : np.ndarray of shape (n,)
    node_names : list of str
    k : int

    Returns
    -------
    top : list of (name, score) tuples
    bottom : list of (name, score) tuples
    """
    ranked = np.argsort(I)[::-1]
    top = [(node_names[i], I[i]) for i in ranked[:k]]
    bottom = [(node_names[i], I[i]) for i in ranked[-k:]]
    return top, bottom
