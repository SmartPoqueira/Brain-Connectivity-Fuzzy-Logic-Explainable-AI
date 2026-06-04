#!/bin/bash
set -e
echo "=== BrainFuzzyXAI: Brain Connectivity Classification ==="
python -m src.model --config configs/config.yaml
echo "=== Done ==="
