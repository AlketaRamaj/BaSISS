"""
Experiment 1 — Spatial patch holdout (within one sample).

Train on 80% of D1 (PD9694d), test on a contiguous spatial patch
of the remaining 20%. Tests spatial interpolation within a tumor.
"""
import sys
sys.path.insert(0, '/Users/D066275/Documents/projectsAImed/BaSISS')

import torch
import numpy as np
from pathlib import Path

from gnn.data_loader import load_patient1
from gnn.model import CloneGAT
from gnn.train_utils import train, full_report, evaluate, plot_prediction_vs_gp, plot_training_history

OUT = Path('/Users/D066275/Documents/projectsAImed/BaSISS/gnn/results/exp1_spatial_patch')
OUT.mkdir(parents=True, exist_ok=True)

CLONE_COLORS = ['#888888','#2ca02c','#9467bd','#1f77b4','#d62728','#ff7f0e','#ffffff','#dddddd']

print("=" * 60)
print("Experiment 1: Spatial patch holdout within D1")
print("=" * 60)

graphs = load_patient1()
g = graphs[0]   # D1
gx, gy = g.grid_shape

print(f"Graph: {gx}x{gy} grid = {gx*gy} nodes, {g.valid.sum().item()} valid")
print(f"Features per node: {g.x.shape[1]}")
print(f"Classes: {g.n_classes}")

# --- Spatial split: hold out the middle horizontal strip (rows 40-70) ---
node_idx = torch.arange(gx * gy)
row_idx   = node_idx // gy   # which row each node is in

test_rows  = (row_idx >= 40) & (row_idx < 70)
train_rows = ~test_rows

train_mask = train_rows & g.valid
test_mask  = test_rows  & g.valid
val_mask   = train_mask   # use train as val for this experiment (small dataset)

print(f"Train nodes: {train_mask.sum().item()} | Test nodes: {test_mask.sum().item()}")

# --- Build and train model ---
torch.manual_seed(42)
model = CloneGAT(
    in_channels     = g.x.shape[1],
    hidden_channels = 64,
    n_classes       = g.n_classes,
    heads=4, dropout=0.3
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nTraining...")
best_val_acc, history = train(model, g, train_mask, val_mask,
                               epochs=300, lr=0.005, patience=40, verbose=True)

# --- Evaluate ---
pred, y_true, y_pred = full_report(model, g, test_mask,
                                    clone_names=g.clone_names, label='Spatial Patch Test')
test_acc, _ = evaluate(model, g, test_mask)
print(f"\nFinal test accuracy: {test_acc:.3f}")

# --- Save outputs ---
plot_training_history(history, OUT / 'training_history.png',
                      title='Exp 1: Spatial Patch Holdout — D1')
plot_prediction_vs_gp(g, pred, OUT / 'prediction_vs_gp.png',
                      clone_names=g.clone_names, clone_colors=CLONE_COLORS,
                      title='Exp 1: GP vs GNN — D1 (middle strip = test)')

torch.save(model.state_dict(), OUT / 'model.pt')
print(f"\nResults saved to {OUT}")
print("Experiment 1 complete.")
