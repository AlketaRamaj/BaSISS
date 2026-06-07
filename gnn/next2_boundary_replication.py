"""
Item 2 — Boundary-distance analysis on PD14780 (replication).

Exact same analysis as boundary_analysis.py but on patient 2.
Uses the Exp2-equivalent models: for each PD14780 sample, use the
model trained on PD9694 (already saved from exp3_cross_patient).
This is a free replication that requires no new training.
"""
import sys, types

pymc = types.ModuleType('pymc'); pymc_gp = types.ModuleType('pymc.gp'); pymc_gp_cov = types.ModuleType('pymc.gp.cov')
class ExpQuad:
    def __init__(self, i, ls): self.ls = ls
    def __call__(self, X):
        import numpy as np; d = X[:,None]-X[None,:]; return np.exp(-0.5*(d/self.ls)**2).squeeze()
pymc_gp_cov.ExpQuad=ExpQuad; pymc_gp.cov=pymc_gp_cov; pymc.gp=pymc_gp; pymc.Beta=object
aesara=types.ModuleType('aesara'); aesara_t=types.ModuleType('aesara.tensor'); aesara_t.gammaln=lambda x:x; aesara.tensor=aesara_t
sys.modules.update({'pymc':pymc,'pymc.gp':pymc_gp,'pymc.gp.cov':pymc_gp_cov,'aesara':aesara,'aesara.tensor':aesara_t})
sys.path.insert(0, '/Users/D066275/Documents/projectsAImed/BaSISS')

import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
from pathlib import Path

from gnn.data_loader import load_patient1, load_patient2
from gnn.model import CloneGAT
from torch_geometric.data import Batch

OUT = Path('/Users/D066275/Documents/projectsAImed/BaSISS/gnn/results/next_steps')
OUT.mkdir(parents=True, exist_ok=True)

GRID_MM = 0.1088
CLONE_COLORS_P2 = ['#1f77b4','#2ca02c','#ff7f0e','#9467bd','#ffffff','#aaaaaa']

def boundary_dist_and_accuracy(graph, model):
    gx, gy = graph.grid_shape
    valid_2d = graph.valid.numpy().reshape(gx, gy)
    dominant = graph.y.numpy().reshape(gx, gy)

    boundary = np.zeros((gx, gy), dtype=bool)
    boundary[:-1,:] |= (dominant[:-1,:] != dominant[1:, :])
    boundary[1:, :] |= (dominant[:-1,:] != dominant[1:, :])
    boundary[:,:-1] |= (dominant[:,:-1] != dominant[:,1:])
    boundary[:,1: ] |= (dominant[:,:-1] != dominant[:,1:])
    boundary |= ~valid_2d
    dist_2d = distance_transform_edt(~boundary) * GRID_MM

    model.eval()
    with torch.no_grad():
        pred = model(graph.x, graph.edge_index).argmax(-1).numpy()

    correct  = (pred == graph.y.numpy()).astype(float)
    valid_f  = graph.valid.numpy()
    dists    = dist_2d.flatten()[valid_f]
    corr_v   = correct[valid_f]

    r, p = spearmanr(dists, corr_v)
    return dists, corr_v, dist_2d, pred, r, p

def bin_accuracy(dists, correct, n_bins=12):
    max_d = np.percentile(dists, 98)
    bins  = np.linspace(0, max_d, n_bins + 1)
    centers = 0.5*(bins[:-1]+bins[1:])
    acc, std = [], []
    for i in range(n_bins):
        m = (dists >= bins[i]) & (dists < bins[i+1])
        if m.sum() > 10:
            acc.append(correct[m].mean())
            std.append(correct[m].std() / np.sqrt(m.sum()))
        else:
            acc.append(np.nan); std.append(np.nan)
    return centers, np.array(acc), np.array(std)

print("Loading PD9694 data (for model input dimensions)...")
graphs_p1 = load_patient1()
in_ch     = graphs_p1[0].x.shape[1]
n_classes = graphs_p1[0].n_classes

print("Loading PD14780 data...")
graphs_p2 = load_patient2()
names_p2  = ['TN1', 'TN2', 'LN1']

# Use the cross-patient model (trained on all PD9694)
print("Loading cross-patient model...")
# For PD14780, train a small model on 2 of its 3 samples, test boundary on 3rd
print("Training PD14780-specific models for boundary analysis...")
from gnn.train_utils import train as train_model

in_ch_p2     = graphs_p2[0].x.shape[1]
n_classes_p2 = graphs_p2[0].n_classes

p2_models = {}
for i, (g_te, name_te) in enumerate(zip(graphs_p2, names_p2)):
    train_graphs_p2 = [g for j, g in enumerate(graphs_p2) if j != i]
    big = Batch.from_data_list(train_graphs_p2)
    m = CloneGAT(in_channels=in_ch_p2, hidden_channels=64,
                 n_classes=n_classes_p2, heads=4, dropout=0.3)
    torch.manual_seed(42)
    train_model(m, big, big.valid, big.valid,
                epochs=200, lr=0.005, patience=30, verbose=False)
    p2_models[name_te] = m
    print(f"  Trained model for {name_te}")

model_for_p2 = p2_models  # dict: name -> model

# Also load PD9694 results for comparison
print("Loading PD9694 models for comparison...")
models_p1 = {}
for name in ['D1','ER1','ER2','D2','D3']:
    m = CloneGAT(in_channels=in_ch, hidden_channels=64,
                 n_classes=n_classes, heads=4, dropout=0.3)
    m.load_state_dict(torch.load(
        f'gnn/results/exp2_leave_one_sample_out/model_{name}.pt', map_location='cpu'))
    models_p1[name] = m

# ── Compute for both patients ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.patch.set_facecolor('black')

all_r, all_p, all_names, all_patients = [], [], [], []

# PD9694
names_p1 = ['D1','ER1','ER2','D2','D3']
TISSUE_P1 = {'D1':'DCIS','ER1':'invasive','ER2':'invasive','D2':'DCIS','D3':'DCIS'}
TISSUE_COLOR = {'DCIS':'#4da6ff','invasive':'#ff4d4d'}

for i, (g, name) in enumerate(zip(graphs_p1, names_p1)):
    if i >= 4: break  # only first 4 fit in plot
    dists, corr, _, _, r, p = boundary_dist_and_accuracy(g, models_p1[name])
    centers, acc, std = bin_accuracy(dists, corr)
    valid = ~np.isnan(acc)
    col   = TISSUE_COLOR[TISSUE_P1[name]]
    ax = axes[0, i]
    ax.fill_between(centers[valid], (acc-std)[valid], (acc+std)[valid],
                    alpha=0.3, color=col)
    ax.plot(centers[valid], acc[valid], color=col, linewidth=2)
    ax.set_title(f'PD9694 {name}\nρ={r:.3f} p={p:.2e}', color='white', fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_facecolor('#111111')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333333')
    if i == 0: ax.set_ylabel('Accuracy', color='white')
    all_r.append(r); all_p.append(p)
    all_names.append(f'PD9694 {name}'); all_patients.append('PD9694')

# PD14780
for i, (g, name) in enumerate(zip(graphs_p2, names_p2)):
    dists, corr, _, _, r, p = boundary_dist_and_accuracy(g, model_for_p2[name])
    centers, acc, std = bin_accuracy(dists, corr)
    valid = ~np.isnan(acc)
    ax = axes[1, i]
    ax.fill_between(centers[valid], (acc-std)[valid], (acc+std)[valid],
                    alpha=0.3, color='#ff9933')
    ax.plot(centers[valid], acc[valid], color='#ff9933', linewidth=2)
    ax.set_title(f'PD14780 {name}\nρ={r:.3f} p={p:.2e}', color='white', fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_facecolor('#111111')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333333')
    ax.set_xlabel('Distance to boundary (mm)', color='white')
    if i == 0: ax.set_ylabel('Accuracy', color='white')
    all_r.append(r); all_p.append(p)
    all_names.append(f'PD14780 {name}'); all_patients.append('PD14780')

# Hide unused subplot
axes[1, 3].set_visible(False)

plt.suptitle('Boundary distance vs accuracy — PD9694 (top) and PD14780 (bottom)\n'
             'Replication across two independent patients',
             color='white', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT / 'item2_boundary_replication.png',
            dpi=120, facecolor='black', bbox_inches='tight')
print("Saved item2_boundary_replication.png")

print("\n" + "="*60)
print("Boundary distance ↔ accuracy — replication summary")
print("="*60)
print(f"{'Sample':<18} {'Patient':<12} {'Spearman ρ':>10} {'p-value':>12}")
print("-"*60)
for name, patient, r, p in zip(all_names, all_patients, all_r, all_p):
    print(f"{name:<18} {patient:<12} {r:>10.3f} {p:>12.2e}")
print("="*60)
print("Positive ρ = accuracy rises with distance from boundary (expected)")
print("Done.")
