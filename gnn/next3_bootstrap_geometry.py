"""
Item 3 — Bootstrap confidence intervals on geometry fingerprints.

With only 5 samples, point estimates of mixing entropy and territory size
are unstable. Bootstrap resampling of tiles within each sample gives
honest error bars.
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
N_BOOT = 500

def mixing_entropy_flat(dom_flat, valid_flat, gx, gy):
    dom = dom_flat.reshape(gx, gy)
    valid = valid_flat.reshape(gx, gy)
    cross = total = 0
    for slc_a, slc_b in [((slice(None,-1),slice(None)),(slice(1,None),slice(None))),
                          ((slice(None),slice(None,-1)),(slice(None),slice(1,None)))]:
        a, b  = dom[slc_a], dom[slc_b]
        va,vb = valid[slc_a], valid[slc_b]
        both  = va & vb
        cross += ((a != b) & both).sum()
        total += both.sum()
    return cross / total if total > 0 else 0

def morans_i_flat(dom_flat, valid_flat, gx, gy):
    dom   = dom_flat.reshape(gx, gy).astype(float)
    valid = valid_flat.reshape(gx, gy)
    vf    = valid.flatten()
    mean_val = dom.flatten()[vf].mean()
    z = np.where(valid, dom - mean_val, 0.0)
    zf = z.flatten()
    W_sum = cross_sum = 0
    for x in range(gx):
        for y in range(gy):
            if not valid[x,y]: continue
            for nx,ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if 0<=nx<gx and 0<=ny<gy and valid[nx,ny]:
                    W_sum += 1; cross_sum += zf[x*gy+y]*zf[nx*gy+ny]
    denom = (zf**2).sum()
    if denom==0 or W_sum==0: return 0.0
    return (vf.sum()/W_sum)*(cross_sum/denom)

def n_territories_flat(dom_flat, valid_flat, gx, gy):
    dom   = dom_flat.reshape(gx, gy)
    valid = valid_flat.reshape(gx, gy)
    count = 0
    for c in range(dom.max()+1):
        _, n = ndimage.label((dom==c) & valid)
        count += n
    return count

print("Loading data..."); graphs = load_patient1()
sample_names = [g.sample_name for g in graphs]

rng = np.random.default_rng(42)
boot_results = {}

print(f"Bootstrapping ({N_BOOT} resamples per sample)...")
for g, name in zip(graphs, sample_names):
    gx, gy  = g.grid_shape
    dom_f   = g.y.numpy()
    valid_f = g.valid.numpy()
    valid_idx = np.where(valid_f)[0]
    n_valid   = len(valid_idx)

    obs = {
        'mixing':  mixing_entropy_flat(dom_f, valid_f, gx, gy),
        'morans':  morans_i_flat(dom_f, valid_f, gx, gy),
        'n_terr':  n_territories_flat(dom_f, valid_f, gx, gy),
    }

    boot = {'mixing': [], 'morans': [], 'n_terr': []}
    for _ in range(N_BOOT):
        # Resample valid tile indices with replacement
        sample_idx = rng.choice(valid_idx, size=n_valid, replace=True)
        boot_dom   = dom_f.copy()
        boot_valid = np.zeros_like(valid_f)
        # Count unique resampled positions (some tiles appear >1, some 0)
        unique_idx, counts = np.unique(sample_idx, return_counts=True)
        boot_valid[unique_idx] = True
        # For multiply-sampled tiles, use observed clone (no change needed)
        bm = mixing_entropy_flat(boot_dom, boot_valid, gx, gy)
        bi = morans_i_flat(boot_dom, boot_valid, gx, gy)
        bn = n_territories_flat(boot_dom, boot_valid, gx, gy)
        boot['mixing'].append(bm)
        boot['morans'].append(bi)
        boot['n_terr'].append(bn)

    ci = {k: np.percentile(boot[k], [2.5, 97.5]) for k in boot}
    boot_results[name] = {'obs': obs, 'ci': ci, 'tissue': TISSUE_TYPE[name]}
    print(f"  {name}: mixing={obs['mixing']:.3f} [{ci['mixing'][0]:.3f},{ci['mixing'][1]:.3f}]  "
          f"morans={obs['morans']:.3f} [{ci['morans'][0]:.3f},{ci['morans'][1]:.3f}]  "
          f"n_terr={obs['n_terr']} [{ci['n_terr'][0]:.0f},{ci['n_terr'][1]:.0f}]")

# ── Plot with CIs ─────────────────────────────────────────────────────────────
metrics = [
    ('mixing', 'Clone mixing entropy'),
    ('morans', "Moran's I"),
    ('n_terr', 'N territories'),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('black')

for ax, (metric, label) in zip(axes, metrics):
    for i, name in enumerate(sample_names):
        r   = boot_results[name]
        col = TISSUE_COLOR[r['tissue']]
        obs = r['obs'][metric]
        lo, hi = r['ci'][metric]
        err_lo = max(0, obs - lo)
        err_hi = max(0, hi - obs)
        ax.bar(i, obs, color=col, alpha=0.8, width=0.6)
        ax.errorbar(i, obs, yerr=[[err_lo],[err_hi]],
                    fmt='none', color='white', capsize=5, linewidth=1.5)
    ax.set_xticks(range(len(sample_names)))
    ax.set_xticklabels(sample_names, color='white', rotation=30)
    ax.set_title(label, color='white')
    ax.set_facecolor('#111111'); ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333333')
    ax.set_ylabel(label, color='white')

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='#4da6ff',label='DCIS'),
                    Patch(color='#ff4d4d',label='invasive')],
           loc='lower center', ncol=2, frameon=False, labelcolor='white',
           bbox_to_anchor=(0.5, -0.05))
plt.suptitle('Geometry fingerprints with 95% bootstrap CIs\n'
             '(resampling valid tiles within each sample)',
             color='white', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'item3_bootstrap_geometry.png',
            dpi=130, facecolor='black', bbox_inches='tight')
print("\nSaved item3_bootstrap_geometry.png")
print("Done.")
