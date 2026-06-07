"""
Load BaSISS data and GP clone labels into PyTorch Geometric graphs.
One graph per tissue sample — nodes are grid squares, edges connect
8-spatial neighbors, features are mutation dot counts + cell density.
"""
import sys, types

# Stub pymc and aesara so basiss imports work without them installed
def _stub_pymc_aesara():
    pymc = types.ModuleType('pymc')
    pymc_gp = types.ModuleType('pymc.gp')
    pymc_gp_cov = types.ModuleType('pymc.gp.cov')
    class ExpQuad:
        def __init__(self, input_dim, ls): self.ls = ls
        def __call__(self, X):
            import numpy as np
            d = X[:,None] - X[None,:]
            return np.exp(-0.5*(d/self.ls)**2).squeeze()
    pymc_gp_cov.ExpQuad = ExpQuad
    pymc_gp.cov = pymc_gp_cov
    pymc.gp = pymc_gp
    pymc.Beta = object
    aesara = types.ModuleType('aesara')
    aesara_t = types.ModuleType('aesara.tensor')
    aesara_t.gammaln = lambda x: x
    aesara.tensor = aesara_t
    sys.modules.update({
        'pymc': pymc, 'pymc.gp': pymc_gp, 'pymc.gp.cov': pymc_gp_cov,
        'aesara': aesara, 'aesara.tensor': aesara_t,
    })

_stub_pymc_aesara()

import numpy as np
import pandas as pd
import cloudpickle as cpkl
import torch
from torch_geometric.data import Data

BASISS_ROOT = '/Users/D066275/Documents/projectsAImed/BaSISS'

# Clone label mappings per patient
CLONE_NAMES_P1 = ['grey', 'green', 'purple', 'blue', 'red', 'orange', 'wt', 'wt2']
CLONE_NAMES_P2 = ['blue', 'green', 'orange', 'purple', 'wt', 'noise']


def _grid_edges(gx, gy):
    """Return edge_index for 8-connected grid of shape (gx, gy)."""
    rows, cols = [], []
    for x in range(gx):
        for y in range(gy):
            src = x * gy + y
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gx and 0 <= ny < gy:
                        rows.append(src)
                        cols.append(nx * gy + ny)
    return torch.tensor([rows, cols], dtype=torch.long)


def _sample_fields(params, sample_idx, n_factors, n_draws=200, seed=42):
    """Sample clone fields from stored GP params."""
    sys.path.insert(0, BASISS_ROOT)
    from basiss.models._parameter_manipulation import sample_essential
    result = sample_essential(params, n_factors=n_factors,
                              samples=[sample_idx], n_draws=n_draws, seed=seed)
    F = result[f'F_{sample_idx}'].mean(0)    # (gx, gy, n_clones)
    lm = result[f'lm_n_{sample_idx}'].mean(0)  # (gx, gy)
    return F, lm


def build_graph(F, lm, dot_counts, mask,
                confidence_thresh=0.25, density_thresh=5.0):
    """
    Build a PyG Data object for one tissue sample.

    Parameters
    ----------
    F           : (gx, gy, n_clones)  clone probability fields
    lm          : (gx, gy)            cell density
    dot_counts  : (gx, gy, n_genes)   raw mutation dot counts
    mask        : (gx, gy)            bool — valid (non-empty) tiles
    """
    gx, gy, n_clones = F.shape
    n_nodes = gx * gy

    # Node features: dot counts + cell density  →  (n_nodes, n_genes+1)
    dot_flat = dot_counts.reshape(n_nodes, -1).astype(np.float32)
    lm_flat  = lm.flatten().astype(np.float32)[:, None]
    x = np.concatenate([dot_flat, lm_flat], axis=1)

    # Labels: dominant clone per node
    dominant   = F.argmax(-1).flatten()        # (n_nodes,)
    confidence = F.max(-1).flatten()           # (n_nodes,)

    # Mask: valid tiles only (enough cells, confident prediction)
    valid = mask.flatten() & (confidence > confidence_thresh) & (lm_flat[:,0] > density_thresh)

    # 8-connected spatial edges
    edge_index = _grid_edges(gx, gy)

    # Grid coordinates (normalised 0-1) as extra positional features
    xs = np.linspace(0, 1, gx)
    ys = np.linspace(0, 1, gy)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    pos = np.stack([XX.flatten(), YY.flatten()], axis=1).astype(np.float32)

    return Data(
        x          = torch.tensor(x, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(dominant, dtype=torch.long),
        pos        = torch.tensor(pos, dtype=torch.float),
        valid      = torch.tensor(valid, dtype=torch.bool),
        confidence = torch.tensor(confidence, dtype=torch.float),
        grid_shape = (gx, gy),
    )


def load_patient1():
    """Load all 5 PD9694 samples as a list of PyG Data objects."""
    print("Loading PD9694 data pkl...")
    with open(f'{BASISS_ROOT}/submission/generated_data/data_structures/data_case1_saved.pkl', 'rb') as f:
        saved = cpkl.load(f)

    print("Loading PD9694 model params...")
    with open(f'{BASISS_ROOT}/submission/generated_data/models/PD9694_bassis_model_params.pkl', 'rb') as f:
        params = cpkl.load(f)

    tree = pd.read_csv(f'{BASISS_ROOT}/data/PD9694_tree.csv', index_col=0)
    genes_to_drop = ['PLXNA2', 'KIF14', 'DSEL']
    tree = tree.iloc[:, ~tree.columns.isin(
        [x+'wt' for x in genes_to_drop] + [x+'mut' for x in genes_to_drop])]
    gene_list = list(tree.columns)

    mut_samples = saved['mut_sample_list']
    scale = 3
    n_factors = 8   # 6 clones + 2 wt

    graphs = []
    sample_names = ['D1', 'ER1', 'ER2', 'D2', 'D3']
    for i, (sample, name) in enumerate(zip(mut_samples, sample_names)):
        print(f"  Building graph for {name} (sample {i})...")
        sample.data_to_grid(scale_factor=scale, probability=0.6)

        # Dot counts grid
        dot_counts = np.stack([sample.gene_grid.get(g, np.zeros_like(
            list(sample.gene_grid.values())[0])) for g in gene_list], axis=-1)

        # Mask
        total = np.array([s for s in sample.gene_grid.values()])[:-3].sum(0)
        infeasible_key = 'infeasible'
        if infeasible_key in sample.gene_grid:
            mask = (sample.gene_grid[infeasible_key] / (total + 1e-8) < 0.1)
        else:
            mask = np.ones(total.shape, dtype=bool)
        mask = mask & (sample.cell_grid > 5)

        # GP clone fields — F shape is (gx, gy, n_factors+n_aug)
        F, lm = _sample_fields(params, i, n_factors)
        n_actual_classes = F.shape[-1]   # includes noise factor

        graph = build_graph(F, lm, dot_counts, mask)
        graph.sample_name   = name
        graph.patient       = 'PD9694'
        graph.clone_names   = CLONE_NAMES_P1
        graph.n_classes     = n_actual_classes
        graphs.append(graph)

    return graphs


def load_patient2():
    """Load all 3 PD14780 samples as a list of PyG Data objects."""
    print("Loading PD14780 data pkl...")
    with open(f'{BASISS_ROOT}/submission/generated_data/data_structures/data_case2_saved.pkl', 'rb') as f:
        saved = cpkl.load(f)

    print("Loading PD14780 model params...")
    with open(f'{BASISS_ROOT}/submission/generated_data/models/PD14780_bassis_model_params.pkl', 'rb') as f:
        params = cpkl.load(f)

    tree = pd.read_csv(f'{BASISS_ROOT}/data/PD14780_tree.csv', index_col=0)
    gene_list = list(tree.columns)

    mut_samples = saved['mut_sample_list']
    scale = 3
    n_factors = len(tree) + 1   # clones + noise

    graphs = []
    sample_names = ['TN1', 'TN2', 'LN1']
    for i, (sample, name) in enumerate(zip(mut_samples, sample_names)):
        print(f"  Building graph for {name} (sample {i})...")
        sample.data_to_grid(scale_factor=scale, probability=0.6)

        dot_counts = np.stack([sample.gene_grid.get(g, np.zeros_like(
            list(sample.gene_grid.values())[0])) for g in gene_list], axis=-1)

        total = np.array([s for s in sample.gene_grid.values()])[:-3].sum(0)
        mask = np.ones(total.shape, dtype=bool)
        mask = mask & (sample.cell_grid > 5)

        F, lm = _sample_fields(params, i, n_factors)
        n_actual_classes = F.shape[-1]

        graph = build_graph(F, lm, dot_counts, mask)
        graph.sample_name   = name
        graph.patient       = 'PD14780'
        graph.clone_names   = CLONE_NAMES_P2
        graph.n_classes     = n_actual_classes
        graphs.append(graph)

    return graphs
