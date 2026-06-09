"""
Training and evaluation utilities shared across all three experiments.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def train_epoch(model, graph, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(graph.x, graph.edge_index)
    loss = F.cross_entropy(logits[train_mask], graph.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, graph, mask):
    model.eval()
    logits = model(graph.x, graph.edge_index)
    pred   = logits.argmax(dim=-1)
    correct = (pred[mask] == graph.y[mask]).sum().item()
    total   = mask.sum().item()
    acc = correct / total if total > 0 else 0.0
    return acc, pred


def train(model, graph, train_mask, val_mask,
          epochs=200, lr=0.005, patience=30, verbose=True):
    """Train with early stopping. Returns best val accuracy and predictions."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=False)

    best_val_acc  = 0.0
    best_state    = None
    no_improve    = 0
    history       = {'train_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, graph, train_mask, optimizer)
        val_acc, _ = evaluate(model, graph, val_mask)
        scheduler.step(1 - val_acc)

        history['train_loss'].append(loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | loss {loss:.4f} | val_acc {val_acc:.3f}"
                  f" | best {best_val_acc:.3f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    return best_val_acc, history


def full_report(model, graph, test_mask, clone_names, label='Test'):
    """Print classification report and return predictions."""
    _, pred = evaluate(model, graph, test_mask)
    y_true = graph.y[test_mask].numpy()
    y_pred = pred[test_mask].numpy()

    present = sorted(set(y_true) | set(y_pred))
    names   = [clone_names[c] if c < len(clone_names) else str(c) for c in present]
    print(f"\n--- {label} Classification Report ---")
    print(classification_report(y_true, y_pred,
                                 labels=present, target_names=names, zero_division=0))
    return pred, y_true, y_pred


def plot_prediction_vs_gp(graph, pred, out_path, clone_names, clone_colors, title=''):
    """Side-by-side: GP labels vs GNN predictions, highlighting disagreements."""
    gx, gy = graph.grid_shape
    valid  = graph.valid.numpy()

    gp_map   = np.full(gx * gy, -1)
    gnn_map  = np.full(gx * gy, -1)
    disagree = np.zeros(gx * gy, dtype=bool)

    gp_map[valid]   = graph.y[valid].numpy()
    gnn_map[valid]  = pred[valid].numpy()
    disagree[valid] = gp_map[valid] != gnn_map[valid]

    gp_map   = gp_map.reshape(gx, gy)
    gnn_map  = gnn_map.reshape(gx, gy)
    disagree = disagree.reshape(gx, gy)

    cmap = mcolors.ListedColormap(['#111111'] + clone_colors[:len(clone_names)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('black')

    for ax, data, t in zip(axes,
                           [gp_map, gnn_map, disagree.astype(int)],
                           ['GP Labels', 'GNN Predictions', 'Disagreements']):
        if t == 'Disagreements':
            ax.imshow(data.T[::-1], cmap='RdYlGn_r', vmin=0, vmax=1,
                      origin='lower', aspect='auto')
        else:
            ax.imshow(data.T[::-1] + 1,
                      cmap=cmap, vmin=0, vmax=len(clone_names),
                      origin='lower', aspect='auto')
        ax.set_title(t, color='white', fontsize=12)
        ax.set_facecolor('black')
        ax.set_xticks([]); ax.set_yticks([])

    pct = disagree[valid.reshape(gx,gy)].mean() * 100
    axes[2].set_xlabel(f'Disagreement rate: {pct:.1f}%', color='white')

    plt.suptitle(title, color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def plot_training_history(history, out_path, title=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('black')
    for ax in (ax1, ax2):
        ax.set_facecolor('#111111')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

    ax1.plot(history['train_loss'], color='#ff7f0e')
    ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')

    ax2.plot(history['val_acc'], color='#2ca02c')
    ax2.set_title('Validation Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')

    plt.suptitle(title, color='white', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")
