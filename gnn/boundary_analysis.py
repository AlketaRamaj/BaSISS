"""
Boundary distance vs prediction accuracy analysis.

For each valid node, compute its distance to the nearest clone boundary
(where the dominant clone changes), then bin nodes by distance and plot
accuracy vs distance. Tests the hypothesis that prediction errors
concentrate at clone boundaries.
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
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

from gnn.data_loader import load_patient1
from gnn.model import CloneGAT
from gnn.train_utils import evaluate

OUT = Path('/Users/D066275/Documents/projectsAImed/BaSISS/gnn/results/boundary_analysis')
OUT.mkdir(parents=True, exist_ok=True)

GRID_MM_PER_TILE = 0.1088   # ~109 µm per grid tile at scale=3

print("Loading data...")
graphs = load_patient1()
sample_names = [g.sample_name for g in graphs]


def compute_boundary_distance(F_mean, valid_mask):
    """
    Compute distance (in tiles) from each node to the nearest clone boundary.

    A boundary exists between any two adjacent tiles with different dominant clones.
    Uses distance_transform_edt on the boundary mask.
    """
    gx, gy = F_mean.shape[:2]
    dominant = F_mean.argmax(-1)   # (gx, gy)

    # A tile is a boundary tile if any of its 4 neighbours has a different dominant clone
    boundary = np.zeros((gx, gy), dtype=bool)
    boundary[:-1, :] |= (dominant[:-1, :] != dominant[1:,  :])
    boundary[1:,  :] |= (dominant[:-1, :] != dominant[1:,  :])
    boundary[:,  :-1] |= (dominant[:, :-1] != dominant[:, 1: ])
    boundary[:, 1:  ] |= (dominant[:, :-1] != dominant[:, 1: ])

    # Also mark invalid tiles as boundaries (tissue edge)
    boundary |= ~valid_mask

    # Distance transform: distance from each pixel to nearest True (boundary) pixel
    dist = distance_transform_edt(~boundary)   # distance to nearest boundary
    return dist, boundary


def accuracy_vs_distance(graph, model, mask, n_bins=15):
    """
    For each node in mask, get its boundary distance and whether the GNN
    prediction was correct. Return binned accuracy curve.
    """
    gx, gy = graph.grid_shape
    valid_2d = graph.valid.numpy().reshape(gx, gy)
    F_mean = graph.confidence.numpy().reshape(gx, gy)   # not F_mean directly

    # Recompute F_mean properly from stored data
    # We stored dominant clone as graph.y and confidence as graph.confidence
    # Reconstruct a proxy dominant map from graph.y
    dominant_2d = graph.y.numpy().reshape(gx, gy)

    # Boundary: adjacent tiles with different dominant clone
    boundary = np.zeros((gx, gy), dtype=bool)
    boundary[:-1, :] |= (dominant_2d[:-1, :] != dominant_2d[1:,  :])
    boundary[1:,  :] |= (dominant_2d[:-1, :] != dominant_2d[1:,  :])
    boundary[:,  :-1] |= (dominant_2d[:, :-1] != dominant_2d[:, 1: ])
    boundary[:, 1:  ] |= (dominant_2d[:, :-1] != dominant_2d[:, 1: ])
    boundary |= ~valid_2d

    dist_2d = distance_transform_edt(~boundary)   # (gx, gy)
    dist_flat = dist_2d.flatten()

    # GNN predictions
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        pred = logits.argmax(-1).numpy()

    correct = (pred == graph.y.numpy()).astype(float)

    # Only consider valid masked nodes
    mask_np = mask.numpy()
    dists   = dist_flat[mask_np]
    corr    = correct[mask_np]

    # Bin by distance
    max_dist = np.percentile(dists, 98)
    bins = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_acc  = []
    bin_std  = []
    bin_n    = []

    for i in range(n_bins):
        in_bin = (dists >= bins[i]) & (dists < bins[i+1])
        if in_bin.sum() > 10:
            bin_acc.append(corr[in_bin].mean())
            bin_std.append(corr[in_bin].std() / np.sqrt(in_bin.sum()))
            bin_n.append(in_bin.sum())
        else:
            bin_acc.append(np.nan)
            bin_std.append(np.nan)
            bin_n.append(0)

    # Pearson and Spearman correlation (node-level)
    r_pearson,  p_pearson  = pearsonr(dists, corr)
    r_spearman, p_spearman = spearmanr(dists, corr)

    return {
        'dist_flat': dists,
        'correct':   corr,
        'bin_centers': bin_centers * GRID_MM_PER_TILE,   # convert to mm
        'bin_acc':   np.array(bin_acc),
        'bin_std':   np.array(bin_std),
        'bin_n':     np.array(bin_n),
        'r_pearson':  r_pearson,
        'p_pearson':  p_pearson,
        'r_spearman': r_spearman,
        'p_spearman': p_spearman,
        'dist_2d':   dist_2d,
        'boundary':  boundary,
        'pred':      pred,
    }


# ── Load trained models ──────────────────────────────────────────────────────
# Exp 1: model trained on D1 (spatial patch split)
model_exp1 = CloneGAT(in_channels=graphs[0].x.shape[1],
                      hidden_channels=64, n_classes=graphs[0].n_classes,
                      heads=4, dropout=0.3)
model_exp1.load_state_dict(torch.load(
    'gnn/results/exp1_spatial_patch/model.pt', map_location='cpu'))

# Exp 2: per-sample models
models_exp2 = {}
for name in sample_names:
    m = CloneGAT(in_channels=graphs[0].x.shape[1],
                 hidden_channels=64, n_classes=graphs[0].n_classes,
                 heads=4, dropout=0.3)
    m.load_state_dict(torch.load(
        f'gnn/results/exp2_leave_one_sample_out/model_{name}.pt', map_location='cpu'))
    models_exp2[name] = m


# ── Analysis ─────────────────────────────────────────────────────────────────
print("\nComputing boundary distance vs accuracy...")

# Exp 1: use the test mask (middle strip rows 40-70)
gx, gy = graphs[0].grid_shape
node_idx = torch.arange(gx * gy)
row_idx  = node_idx // gy
test_mask_exp1 = ((row_idx >= 40) & (row_idx < 70)) & graphs[0].valid

results = {}
results['D1_exp1'] = accuracy_vs_distance(graphs[0], model_exp1, test_mask_exp1)
print(f"  D1 (Exp1): r={results['D1_exp1']['r_pearson']:.3f}, "
      f"ρ={results['D1_exp1']['r_spearman']:.3f}")

for i, (g, name) in enumerate(zip(graphs, sample_names)):
    mask = g.valid
    res = accuracy_vs_distance(g, models_exp2[name], mask)
    results[name] = res
    print(f"  {name} (Exp2): r={res['r_pearson']:.3f}, ρ={res['r_spearman']:.3f}")


# ── Main plot: accuracy vs boundary distance ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('black')
axes = axes.flatten()

plot_items = [('D1_exp1', 'D1 — Exp1 (spatial patch)', graphs[0])] + \
             [(name, f'{name} — Exp2 (leave-one-out)', graphs[i])
              for i, name in enumerate(sample_names)]

for ax, (key, title, g) in zip(axes, plot_items):
    res = results[key]
    bc  = res['bin_centers']
    ba  = res['bin_acc']
    bs  = res['bin_std']
    valid = ~np.isnan(ba)

    ax.fill_between(bc[valid], (ba-bs)[valid], (ba+bs)[valid],
                    alpha=0.3, color='#2ca02c')
    ax.plot(bc[valid], ba[valid], color='#2ca02c', linewidth=2)
    ax.axhline(y=ba[valid].mean(), color='white', linestyle='--',
               linewidth=0.8, alpha=0.5, label=f'mean {ba[valid].mean():.2f}')

    ax.set_xlabel('Distance to nearest clone boundary (mm)', color='white')
    ax.set_ylabel('Prediction accuracy', color='white')
    ax.set_title(f'{title}\nr={res["r_pearson"]:.3f}  ρ={res["r_spearman"]:.3f}',
                 color='white', fontsize=10)
    ax.set_facecolor('#111111')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#444444')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, labelcolor='white', fontsize=8)

plt.suptitle('Boundary Distance vs Prediction Accuracy\n'
             'Hypothesis: accuracy increases with distance from clone boundary',
             color='white', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT / 'accuracy_vs_boundary_distance.png',
            dpi=150, facecolor='black', bbox_inches='tight')
print(f"\nSaved accuracy_vs_boundary_distance.png")


# ── Spatial map: distance field overlaid with errors ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('black')

res = results['D1_exp1']
gx, gy = graphs[0].grid_shape
dist_2d = res['dist_2d'] * GRID_MM_PER_TILE

correct_2d = np.full(gx * gy, np.nan)
mask_np = test_mask_exp1.numpy()
correct_2d[mask_np] = res['correct']
correct_2d = correct_2d.reshape(gx, gy)

im0 = axes[0].imshow(dist_2d.T[::-1], cmap='magma', origin='lower', aspect='auto')
axes[0].set_title('Distance to clone boundary (mm)', color='white')
plt.colorbar(im0, ax=axes[0], label='mm').ax.yaxis.set_tick_params(color='white')

# Error map: red=wrong, green=correct, grey=train
error_rgba = np.zeros((gx, gy, 4))
error_rgba[correct_2d == 1] = [0.17, 0.63, 0.17, 0.9]   # green = correct
error_rgba[correct_2d == 0] = [0.84, 0.15, 0.15, 0.9]   # red = wrong
axes[1].imshow(error_rgba.transpose(1,0,2)[::-1], origin='lower', aspect='auto')
axes[1].set_title('GNN errors on test strip\n(green=correct, red=wrong)', color='white')

for ax in axes:
    ax.set_facecolor('black'); ax.tick_params(colors='white')
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('D1 — Spatial pattern of errors vs boundary proximity',
             color='white', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'spatial_error_map_D1.png',
            dpi=150, facecolor='black', bbox_inches='tight')
print("Saved spatial_error_map_D1.png")


# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Boundary distance ↔ accuracy correlation summary")
print("="*50)
print(f"{'Sample':<15} {'Pearson r':>10} {'Spearman ρ':>12} {'p-value':>12}")
print("-"*50)
for key, res in results.items():
    print(f"{key:<15} {res['r_pearson']:>10.3f} {res['r_spearman']:>12.3f} "
          f"{res['p_pearson']:>12.2e}")
print("="*50)
print("\nDone.")
