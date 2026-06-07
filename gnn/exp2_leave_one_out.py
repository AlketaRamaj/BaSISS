"""
Experiment 2 — Leave-one-sample-out (within patient 1).

Train on 4 of 5 PD9694 samples, test on the held-out sample.
Repeat for each of the 5 samples. Tests generalisation to unseen
tissue within the same patient.

Note: Patient 2 is left out: To do it properly on PD14780 we would need at least 4–5 samples with balanced clone
  representation and fewer noise-labelled tiles.
"""
import sys
sys.path.insert(0, '/Users/D066275/Documents/projectsAImed/BaSISS')

import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Batch

from gnn.data_loader import load_patient1
from gnn.model import CloneGAT
from gnn.train_utils import train, full_report, evaluate, plot_prediction_vs_gp, plot_training_history

OUT = Path('/Users/D066275/Documents/projectsAImed/BaSISS/gnn/results/exp2_leave_one_sample_out')
OUT.mkdir(parents=True, exist_ok=True)

CLONE_COLORS = ['#888888','#2ca02c','#9467bd','#1f77b4','#d62728','#ff7f0e','#ffffff','#dddddd']

print("=" * 60)
print("Experiment 2: Leave-one-sample-out within PD9694")
print("=" * 60)

graphs = load_patient1()
sample_names = [g.sample_name for g in graphs]
results = {}

for held_out_idx in range(len(graphs)):
    held_out_name = sample_names[held_out_idx]
    print(f"\n--- Holding out: {held_out_name} ---")

    test_graph  = graphs[held_out_idx]
    train_graphs = [graphs[i] for i in range(len(graphs)) if i != held_out_idx]

    # Merge training graphs into one large graph via Batch
    # (disconnected — no edges across samples, which is correct)
    big_graph = Batch.from_data_list(train_graphs)
    train_mask = big_graph.valid
    val_mask   = train_mask

    test_mask = test_graph.valid

    print(f"  Train nodes: {train_mask.sum().item()} across {len(train_graphs)} samples")
    print(f"  Test nodes:  {test_mask.sum().item()} in {held_out_name}")

    torch.manual_seed(42)
    model = CloneGAT(
        in_channels     = big_graph.x.shape[1],
        hidden_channels = 64,
        n_classes       = graphs[0].n_classes,
        heads=4, dropout=0.3
    )

    best_val, history = train(model, big_graph, train_mask, val_mask,
                               epochs=300, lr=0.005, patience=40, verbose=True)

    pred, y_true, y_pred = full_report(model, test_graph, test_mask,
                                        clone_names=test_graph.clone_names,
                                        label=f'Test: {held_out_name}')
    test_acc, _ = evaluate(model, test_graph, test_mask)
    results[held_out_name] = test_acc
    print(f"  Test accuracy: {test_acc:.3f}")

    plot_training_history(history,
                          OUT / f'history_{held_out_name}.png',
                          title=f'Exp 2: Hold out {held_out_name}')
    plot_prediction_vs_gp(test_graph, pred,
                           OUT / f'pred_vs_gp_{held_out_name}.png',
                           clone_names=test_graph.clone_names,
                           clone_colors=CLONE_COLORS,
                           title=f'Exp 2: GP vs GNN — held out {held_out_name}')
    torch.save(model.state_dict(), OUT / f'model_{held_out_name}.pt')

# Summary
print("\n" + "=" * 40)
print("Experiment 2 Summary")
print("=" * 40)
for name, acc in results.items():
    print(f"  {name}: {acc:.3f}")
print(f"  Mean: {np.mean(list(results.values())):.3f}")
print(f"\nResults saved to {OUT}")
print("Experiment 2 complete.")
