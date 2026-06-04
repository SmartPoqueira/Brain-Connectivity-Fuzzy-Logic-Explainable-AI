# BrainFuzzyXAI: Brain Connectivity Classification with Fuzzy Logic and Explainable AI

Classification of **preterm vs. term infant brain structural connectivity** using machine learning and graph neural networks, enhanced with **fuzzy logic label smoothing** and **SHAP explainability**. Published at ECAI 2025 (CEUR Workshop Proceedings).

## Overview

Preterm birth disrupts critical neurodevelopmental processes during the final trimester. This project uses structural connectomes (DTI-derived 90×90 adjacency matrices) from the Developing Human Connectome Project (dHCP) to classify infants as preterm or term. Key contributions:

1. **Spatial coordinate augmentation** — Atlas-based centroid coordinates as node features improve LR accuracy from 88.6% to 93.3%.
2. **Fuzzy logic label smoothing** — A sigmoidal soft target around the 37-week threshold: $y_i^{\text{soft}} = \sigma\left(\frac{GA_i - \tau}{T}\right)$, achieving **96.2% accuracy**.
3. **SHAP explainability** — Edge-importance matrices and node-level aggregation identify key brain regions (thalamus, putamen, cingulum) consistent with neuroscience literature.

## Method

### Data Representation

Each brain is represented as:
- **Matrix view**: Symmetric adjacency matrix $A \in \mathbb{R}^{90 \times 90}$ → upper-triangle feature vector of $d = 4{,}005$ edge weights.
- **Graph view**: Weighted undirected graph $G = (V, E, X_v, X_e)$ with atlas-based spatial coordinates as node features.

### Fuzzy Logic

The standard hard label at 37 weeks is replaced with:

$$y_i^{\text{soft}} = \frac{1}{1 + \exp\left(-\frac{GA_i - 37}{T}\right)}$$

where $T$ controls transition smoothness. This handles dating uncertainty and the biological continuum around the preterm/term boundary.

### Models

| Model | Type | Best Accuracy |
|---|---|---|
| Logistic Regression + Spatial | Matrix-based | 93.3% |
| LR + Spatial + Fuzzy | Matrix-based | **96.2%** |
| GAT | Graph-based | 90.0% |
| GAT + Fuzzy | Graph-based | 90.0% |
| GCN | Graph-based | 89.0% |

## Results

Best model: **LR + Spatial Coordinates + Fuzzy Logic**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Preterm | 0.95 | 0.86 | 0.90 |
| Term | 0.97 | 0.99 | 0.98 |
| **Overall (weighted)** | **0.96** | **0.96** | **0.96** |

### SHAP Explainability

Top-10 most important brain regions (consistent across LR and GAT):
- **Thalamus R**, **Putamen R**, **Insula R**, **Frontal Mid L**

Bottom-10 (least discriminative — early-maturing regions):
- **Heschl L/R**, **Occipital Sup L**, **Paracentral Lob R**, **Palladium L**

## Project Structure

```
BrainFuzzyXAI/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── model.py              # ML + GNN classifiers
│   ├── fuzzy.py              # Fuzzy label smoothing
│   ├── explainability.py     # SHAP analysis pipeline
│   ├── data_loader.py        # dHCP data loader
│   └── spatial.py            # Atlas coordinate extraction
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
└── scripts/
    └── run_experiment.sh
```

## Data

Data from the [Developing Human Connectome Project (dHCP)](http://www.developingconnectome.org/), processed by [Taoudi-Benchekroun et al.](https://github.com/CoDe-Neuro/Predicting-age-and-clinical-risk-from-the-neonatal-connectome). 524 infant structural connectomes (90 brain regions).

## Quick Start

```bash
pip install -r requirements.txt
python -m src.model --config configs/config.yaml
```

## Citation

```bibtex
@inproceedings{birch2025exploring,
  title={Exploring Structural Brain Connectivity in Term and Preterm Infants with Explainable AI and Fuzzy Logic},
  author={Birch, Katherine and Dur{\'a}n-L{\'o}pez, Alberto and Bola{\~n}os-Mart{\'i}nez, Daniel and Pravin, Chandresh and Berm{\'u}dez-Edo, Mar{\'i}a and Bauer, Roman and De, Suparna},
  booktitle={ECAI 2025 Workshop Proceedings},
  year={2025},
  organization={CEUR Workshop Proceedings}
}
```

## License

MIT License — see [LICENSE](LICENSE).
