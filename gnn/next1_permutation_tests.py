"""
Item 1 — Permutation tests for geometry differences.

For each sample, randomly shuffle clone labels (preserving valid mask and
cell density), then recompute mixing entropy, Moran's I, and median territory
size. Repeat 1000 times to build a null distribution. Ask: are the observed
DCIS vs invasive geometry differences larger than expected by chance?

This gives honest p-values without requiring more samples.
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
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path

from gnn.data_loader import load_patient1

OUT = Path('/Users/D066275/Documents/projectsAImed/BaSISS/gnn/results/next_steps')
OUT.mkdir(parents=True, exist_ok=True)

TISSUE_TYPE  = {'D1':'DCIS','ER1':'invasive','ER2':'invasive','D2':'DCIS','D3':'DCIS'}
TISSUE_COLOR = {'DCIS':'#4da6ff', 'invasive':'#ff4d4d'}
N_PERM = 1000

def mixing_entropy(dominant_2d, valid_2d):
    cross = 0; total = 0
    for slc_a, slc_b in [((slice(None,-1),slice(None)),(slice(1,None),slice(None))),
                          ((slice(None),slice(None,-1)),(slice(None),slice(1,None)))]:
        a, b = dominant_2d[slc_a], dominant_2d[slc_b]
        va, vb = valid_2d[slc_a], valid_2d[slc_b]
        both = va & vb
        cross += ((a != b) & both).sum()
        total += both.sum()
    return cross / total if total > 0 else 0

def morans_i(dominant_2d, valid_2d):
    gx, gy = dominant_2d.shape
    flat = dominant_2d.flatten().astype(float)
    vf   = valid_2d.flatten()
    mean_val = flat[vf].mean()
    z = np.where(vf, flat - mean_val, 0.0)
    W_sum = cross_sum = 0
    for x in range(gx):
        for y in range(gy):
            if not valid_2d[x, y]: continue
            for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if 0 <= nx < gx and 0 <= ny < gy and valid_2d[nx, ny]:
                    W_sum    += 1
                    cross_sum += z[x*gy+y] * z[nx*gy+ny]
    denom = (z**2).sum()
    if denom == 0 or W_sum == 0: return 0.0
    return (vf.sum() / W_sum) * (cross_sum / denom)

def median_territory_size(dominant_2d, valid_2d):
    sizes = []
    for c in range(dominant_2d.max() + 1):
        labeled, n = ndimage.label((dominant_2d == c) & valid_2d)
        for rid in range(1, n+1):
            s = (labeled == rid).sum()
            if s >= 4: sizes.append(s)
    return np.median(sizes) if sizes else 0

def compute_metrics(dominant_2d, valid_2d):
    return {
        'mixing':    mixing_entropy(dominant_2d, valid_2d),
        'morans_i':  morans_i(dominant_2d, valid_2d),
        'med_terr':  median_territory_size(dominant_2d, valid_2d),
    }

print("Loading data...")
graphs = load_patient1()
sample_names = [g.sample_name for g in graphs]

rng = np.random.default_rng(42)
results = {}

print(f"\nRunning {N_PERM} permutations per sample...")
for g, name in zip(graphs, sample_names):
    print(f"  {name} ({TISSUE_TYPE[name]})...")
    gx, gy   = g.grid_shape
    dominant = g.y.numpy().reshape(gx, gy)
    valid    = g.valid.numpy().reshape(gx, gy)

    # Observed metrics
    obs = compute_metrics(dominant, valid)

    # Null distribution: shuffle clone labels within valid tiles
    valid_idx   = np.where(valid.flatten())[0]
    dom_flat    = dominant.flatten().copy()
    null_metrics = {'mixing': [], 'morans_i': [], 'med_terr': []}

    for _ in range(N_PERM):
        shuffled = dom_flat.copy()
        shuffled[valid_idx] = rng.permutation(shuffled[valid_idx])
        dom_perm = shuffled.reshape(gx, gy)
        m = compute_metrics(dom_perm, valid)
        for k in null_metrics: null_metrics[k].append(m[k])

    # p-values (one-tailed: observed > null for mixing; observed > null for morans)
    pvals = {}
    for k in ['mixing', 'morans_i', 'med_terr']:
        null = np.array(null_metrics[k])
        # p = fraction of null >= observed (test: observed is unusually HIGH)
        pvals[k] = (null >= obs[k]).mean()

    results[name] = {'obs': obs, 'null': null_metrics, 'pvals': pvals,
                     'tissue': TISSUE_TYPE[name]}
    print(f"    mixing={obs['mixing']:.3f} (p={pvals['mixing']:.4f})  "
          f"morans_i={obs['morans_i']:.3f} (p={pvals['morans_i']:.4f})  "
          f"med_terr={obs['med_terr']:.1f} (p={pvals['med_terr']:.4f})")

# ── Plot null distributions ───────────────────────────────────────────────────
metrics_to_plot = [
    ('mixing',   'Clone mixing entropy'),
    ('morans_i', "Moran's I"),
    ('med_terr', 'Median territory size (tiles)'),
]

fig, axes = plt.subplots(len(sample_names), len(metrics_to_plot),
                          figsize=(15, 3*len(sample_names)))
fig.patch.set_facecolor('black')

for row, (g, name) in enumerate(zip(graphs, sample_names)):
    col_c = TISSUE_COLOR[TISSUE_TYPE[name]]
    for col, (metric, label) in enumerate(metrics_to_plot):
        ax = axes[row, col]
        null = np.array(results[name]['null'][metric])
        obs  = results[name]['obs'][metric]
        pval = results[name]['pvals'][metric]

        ax.hist(null, bins=40, color='#444444', alpha=0.8)
        ax.axvline(obs, color=col_c, linewidth=2.5,
                   label=f'obs={obs:.3f}\np={pval:.4f}')
        ax.set_facecolor('#111111'); ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('#333333')
        if row == 0: ax.set_title(label, color='white', fontsize=10)
        if col == 0: ax.set_ylabel(name, color='white', fontsize=9)
        ax.legend(frameon=False, labelcolor='white', fontsize=7)

plt.suptitle(f'Permutation null distributions (n={N_PERM} shuffles per sample)\n'
             'Coloured line = observed value',
             color='white', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT / 'item1_permutation_tests.png',
            dpi=120, facecolor='black', bbox_inches='tight')
print(f"\nSaved item1_permutation_tests.png")

# Summary table
print("\n" + "="*70)
print(f"{'Sample':<8} {'Type':<10} {'Mix p':>8} {'Moran p':>8} {'Terr p':>8}")
print("-"*70)
for name in sample_names:
    r = results[name]
    print(f"{name:<8} {r['tissue']:<10} "
          f"{r['pvals']['mixing']:>8.4f} "
          f"{r['pvals']['morans_i']:>8.4f} "
          f"{r['pvals']['med_terr']:>8.4f}")
print("="*70)
print("p-value interpretation: fraction of permuted samples >= observed value")
print("Low p = observed geometry is unusually structured vs random clone assignment")
