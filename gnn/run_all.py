"""
run_all.py — single entry point to reproduce all GNN experiments.

Run from the repo root:
    python gnn/run_all.py

Runs in sequence:
    1. Experiment 1 — spatial patch holdout (within D1)
    2. Experiment 2 — leave-one-sample-out (all 5 folds, PD9694)
    3. Boundary-distance analysis (PD9694)
    4. Geometry fingerprints + permutation tests + bootstrap CIs
    5. Boundary-distance replication on PD14780
    6. GAT attention-weight visualisation
    7. Falsification of UMAP AUC=1.0

All figures → gnn/results/
Expected wall time: ~15 min CPU, ~5 min GPU.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPTS = [
    ROOT / 'gnn' / 'exp1_spatial_patch.py',
    ROOT / 'gnn' / 'exp2_leave_one_out.py',
    ROOT / 'gnn' / 'boundary_analysis.py',
    ROOT / 'gnn' / 'next1_permutation_tests.py',
    ROOT / 'gnn' / 'next2_boundary_replication.py',
    ROOT / 'gnn' / 'next3_bootstrap_geometry.py',
    ROOT / 'gnn' / 'next4_attention_weights.py',
    ROOT / 'gnn' / 'falsify_auc.py',
]

NAMES = [
    'Exp 1 — spatial patch holdout',
    'Exp 2 — leave-one-sample-out',
    'Boundary-distance analysis (PD9694)',
    'Geometry permutation tests',
    'Boundary-distance replication (PD14780)',
    'Bootstrap geometry CIs',
    'GAT attention weights',
    'Falsification of AUC=1.0',
]


def run(script: Path, name: str) -> bool:
    print(f'\n{"=" * 60}')
    print(f'  {name}')
    print(f'  {script.name}')
    print(f'{"=" * 60}')
    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)], check=False)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = 'OK' if ok else f'FAILED (exit {result.returncode})'
    print(f'\n  → {status}  ({elapsed:.1f}s)')
    return ok


if __name__ == '__main__':
    results = {}
    for script, name in zip(SCRIPTS, NAMES):
        if not script.exists():
            print(f'\nSKIP (not found): {script}')
            results[name] = 'SKIP'
            continue
        ok = run(script, name)
        results[name] = 'OK' if ok else 'FAILED'

    print(f'\n{"=" * 60}')
    print('  Summary')
    print(f'{"=" * 60}')
    for name, status in results.items():
        tag = '✓' if status == 'OK' else ('?' if status == 'SKIP' else '✗')
        print(f'  {tag}  {name}: {status}')

    failed = [n for n, s in results.items() if s == 'FAILED']
    if failed:
        print(f'\n{len(failed)} script(s) failed.')
        sys.exit(1)
    else:
        print('\nAll experiments complete.')
        print('Figures written to gnn/results/')
