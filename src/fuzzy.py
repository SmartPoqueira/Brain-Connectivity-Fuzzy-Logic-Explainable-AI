"""
Fuzzy logic label smoothing for gestational age classification.

Replaces hard binary labels (preterm/term at 37 weeks) with soft targets
using a sigmoidal mapping, reflecting the continuous nature of brain development.
"""

import numpy as np


def fuzzy_labels(ga_values, tau=37.0, T=1.0):
    """Compute fuzzy soft targets from gestational age values.

    Parameters
    ----------
    ga_values : array-like of shape (n_samples,)
        Gestational age at birth in weeks.
    tau : float
        Decision threshold (default: 37 weeks).
    T : float
        Temperature parameter controlling transition smoothness.
        Lower T = sharper boundary. Higher T = smoother transition.

    Returns
    -------
    y_soft : np.ndarray of shape (n_samples,)
        Soft targets in [0, 1]. Values near 0 indicate preterm,
        values near 1 indicate term.
    """
    ga = np.asarray(ga_values, dtype=np.float64)
    y_soft = 1.0 / (1.0 + np.exp(-(ga - tau) / T))
    return y_soft


def hard_labels(ga_values, tau=37.0):
    """Standard hard binary labeling.

    Parameters
    ----------
    ga_values : array-like
        Gestational age at birth in weeks.
    tau : float
        Threshold. GA < tau → preterm (0), GA >= tau → term (1).

    Returns
    -------
    y : np.ndarray of int
    """
    ga = np.asarray(ga_values, dtype=np.float64)
    return (ga >= tau).astype(int)
