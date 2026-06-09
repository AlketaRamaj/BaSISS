"""
Item 4 — GAT attention weight visualisation.

The GAT model produces per-edge attention weights during forward pass.
For each node, we have a weight for each of its up-to-8 neighbours.

We ask:
  1. Do boundary nodes attend differently than interior nodes?
  2. Do nodes attend more strongly to same-clone neighbours?
  3. Is attention entropy (spread across neighbours) higher at boundaries?

This makes Finding 1 interpretable without any classifier.
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
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr, mannwhitneyu
from pathlib import Path

from gnn.data_loader import load_patient1
from gnn.model import CloneGAT

OUT = Path('/Users/D066275/Documents/projectsAImed/BaSISS/gnn/results/next_steps')
OUT.mkdir(parents=True, exist_ok=True)

GRID_MM = 0.1088
TISSUE_TYPE = {'D1':'DCIS','ER1':'invasive','ER2':'invasive','D2':'DCIS','D3':'DCIS'}

def get_attention_weights(model, graph):
    """
    Extract per-node attention entropy and same-clone attention fraction
    from the first GAT layer.
    Returns:
      attn_entropy  : (n_nodes,) — how spread is attention across neighbours
      same_clone_attn: (n_nodes,) — fraction of attention going to same-clone neighbours
    """
    model.eval()
    hooks = {}
    attention_store = []

    # Hook into conv1 to capture attention coefficients
    def hook_fn(module, input, output):
        # GATConv returns (out, attn_weights) when return_attention_weights=True
        attention_store.clear()
        attention_store.append(output)

    with torch.no_grad():
        # Call conv1 with return_attention_weights=True
        x = graph.x
        ei = graph.edge_index
        out, (edge_idx, attn) = model.conv1(x, ei, return_attention_weights=True)
        # attn: (n_edges, n_heads)  — average across heads
        attn_mean = attn.mean(dim=-1).numpy()  # (n_edges,)

        src, dst = edge_idx.numpy()
        n_nodes  = x.shape[0]
        clones   = graph.y.numpy()

        # Per-node: attention entropy (higher = spreads evenly across neighbours)
        attn_entropy    = np.zeros(n_nodes)
        same_clone_attn = np.zeros(n_nodes)
        n_edges_per_node = np.zeros(n_nodes)

        for e_idx in range(len(src)):
            d = dst[e_idx]
            s = src[e_idx]
            w = attn_mean[e_idx]
            n_edges_per_node[d] += 1

        # Compute entropy per destination node
        for d in range(n_nodes):
            edge_mask = dst == d
            if edge_mask.sum() == 0: continue
            weights = attn_mean[edge_mask]
            sources = src[edge_mask]
            # Normalise (should already sum to ~1 per dst node after softmax)
            weights = weights / (weights.sum() + 1e-8)
            # Entropy: -sum(w * log(w))
            attn_entropy[d] = -np.sum(weights * np.log(weights + 1e-8))
            # Same-clone fraction
            same = clones[sources] == clones[d]
            same_clone_attn[d] = weights[same].sum()

    return attn_entropy, same_clone_attn


print("Loading data...")
graphs = load_patient1()
sample_names = [g.sample_name for g in graphs]

fig, axes = plt.subplots(3, 5, figsize=(22, 12))
fig.patch.set_facecolor('black')

all_stats = []
for col_i, (g, name) in enumerate(zip(graphs, sample_names)):
    model = CloneGAT(in_channels=g.x.shape[1], hidden_channels=64,
                     n_classes=g.n_classes, heads=4, dropout=0.3)
    model.load_state_dict(torch.load(
        f'gnn/results/exp2_leave_one_sample_out/model_{name}.pt', map_location='cpu'))

    print(f"  {name}...")
    attn_ent, same_attn = get_attention_weights(model, g)

    gx, gy  = g.grid_shape
    valid_2d = g.valid.numpy().reshape(gx, gy)
    dominant = g.y.numpy().reshape(gx, gy)

    # Boundary mask
    boundary = np.zeros((gx, gy), dtype=bool)
    boundary[:-1,:] |= (dominant[:-1,:] != dominant[1:, :])
    boundary[1:, :] |= (dominant[:-1,:] != dominant[1:, :])
    boundary[:,:-1] |= (dominant[:,:-1] != dominant[:,1:])
    boundary[:,1: ] |= (dominant[:,:-1] != dominant[:,1:])
    boundary |= ~valid_2d
    dist_2d = distance_transform_edt(~boundary) * GRID_MM
    is_boundary = (dist_2d < GRID_MM * 1.5) & valid_2d  # within 1.5 tiles of boundary

    valid_f   = g.valid.numpy()
    dist_f    = dist_2d.flatten()[valid_f]
    ent_f     = attn_ent[valid_f]
    same_f    = same_attn[valid_f]
    bound_f   = is_boundary.flatten()[valid_f]

    # Spearman: dist vs attention entropy
    r_ent, p_ent = spearmanr(dist_f, ent_f)
    # Spearman: dist vs same-clone attention
    r_same, p_same = spearmanr(dist_f, same_f)
    # Mann-Whitney: boundary vs interior entropy
    stat, p_mw = mannwhitneyu(ent_f[bound_f], ent_f[~bound_f], alternative='greater')

    all_stats.append({'name': name, 'tissue': TISSUE_TYPE[name],
                      'r_ent': r_ent, 'p_ent': p_ent,
                      'r_same': r_same, 'p_same': p_same,
                      'p_mw': p_mw,
                      'mean_ent_boundary': ent_f[bound_f].mean(),
                      'mean_ent_interior': ent_f[~bound_f].mean()})

    # Row 0: attention entropy map
    ent_2d = np.full(gx * gy, np.nan)
    ent_2d[valid_f] = ent_f
    ent_2d = ent_2d.reshape(gx, gy)
    ax = axes[0, col_i]
    im = ax.imshow(ent_2d.T[::-1], cmap='plasma', origin='lower', aspect='auto')
    ax.set_title(f'{name}\nent ρ={r_ent:.2f}', color='white', fontsize=9)
    ax.set_facecolor('black'); ax.set_xticks([]); ax.set_yticks([])
    if col_i == 0: ax.set_ylabel('Attn entropy', color='white')

    # Row 1: same-clone attention map
    same_2d = np.full(gx * gy, np.nan)
    same_2d[valid_f] = same_f
    same_2d = same_2d.reshape(gx, gy)
    ax = axes[1, col_i]
    ax.imshow(same_2d.T[::-1], cmap='RdYlGn', vmin=0, vmax=1,
              origin='lower', aspect='auto')
    ax.set_title(f'same-clone attn\nρ={r_same:.2f}', color='white', fontsize=9)
    ax.set_facecolor('black'); ax.set_xticks([]); ax.set_yticks([])
    if col_i == 0: ax.set_ylabel('Same-clone attn', color='white')

    # Row 2: boundary vs interior entropy violin
    ax = axes[2, col_i]
    parts = ax.violinplot([ent_f[bound_f], ent_f[~bound_f]],
                           positions=[0, 1], showmedians=True)
    for pc, col in zip(parts['bodies'], ['#ff4d4d','#4da6ff']):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    ax.set_xticks([0,1]); ax.set_xticklabels(['boundary','interior'],
                                               color='white', fontsize=8)
    ax.set_title(f'MW p={p_mw:.2e}', color='white', fontsize=8)
    ax.set_facecolor('#111111'); ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333333')
    if col_i == 0: ax.set_ylabel('Attention entropy', color='white')

plt.suptitle('GAT Attention Weight Analysis\n'
             'Row 1: entropy map  |  Row 2: same-clone attention  |  Row 3: boundary vs interior',
             color='white', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT / 'item4_attention_weights.png',
            dpi=120, facecolor='black', bbox_inches='tight')
print("\nSaved item4_attention_weights.png")

print("\n" + "="*70)
print(f"{'Sample':<8} {'Tissue':<10} {'ρ(dist,ent)':>11} {'ρ(dist,same)':>13} {'MannWhitney p':>14}")
print("-"*70)
for s in all_stats:
    print(f"{s['name']:<8} {s['tissue']:<10} {s['r_ent']:>11.3f} "
          f"{s['r_same']:>13.3f} {s['p_mw']:>14.2e}")
    print(f"  mean entropy: boundary={s['mean_ent_boundary']:.3f}  "
          f"interior={s['mean_ent_interior']:.3f}")
print("="*70)
print("ρ(dist, entropy): positive = further from boundary → lower entropy (more focused)")
print("ρ(dist, same):    positive = further from boundary → more same-clone attention")
print("MannWhitney p:    boundary nodes have higher attention entropy than interior")
print("Done.")
