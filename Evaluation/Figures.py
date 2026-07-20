import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.patches as mpatches

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ['#08306B', '#F7FBFF']
    for r in range(2):
        for c in range(2):
            color = colors[0] if r == c else colors[1]
            txt_color = 'white' if r == c else 'black'
            rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color,
                                      edgecolor='black', linewidth=3)
            ax.add_patch(rect)
            ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                    fontsize=24, fontweight='bold', color=txt_color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Depressive', 'Non-Depressive'], fontsize=18, fontweight='bold')
    ax.set_yticklabels(['Depressive', 'Non-Depressive'], fontsize=18, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=20, fontweight='bold')
    ax.set_ylabel('True', fontsize=20, fontweight='bold')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(length=0)
    ax.set_aspect('equal')
    plt.title(f'{model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_roc_curves(all_y_true, all_y_prob, model_names, save_path):
    plt.figure(figsize=(8, 6))
    for name, yt, yp in zip(model_names, all_y_true, all_y_prob):
        fpr, tpr, _ = roc_curve(yt, yp)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_loss_acc_curves(history, model_name, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2.3, label='Training Loss')
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2.3, label='Validation Loss')
    ax.set_title(f'{model_name} Loss', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Loss', fontsize=18)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_loss.png', dpi=600)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(epochs, history['train_acc'], 'b-', linewidth=2.3, label='Training Accuracy')
    ax.plot(epochs, history['val_acc'], 'r-', linewidth=2.3, label='Validation Accuracy')
    ax.set_title(f'{model_name} Accuracy', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Accuracy', fontsize=18)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_acc.png', dpi=600)
    plt.close()

def _radar_limits(metric_name):
    if metric_name in ['Accuracy', 'Precision', 'Recall', 'Specificity', 'NPV', 'F1']:
        return (0, 100), np.arange(0, 101, 20), [str(x) for x in range(0, 101, 20)]
    elif metric_name == 'Time (s)':
        return (0, 60), np.arange(0, 61, 10), [str(x) for x in range(0, 61, 10)]
    elif metric_name in ['FPR', 'FNR']:
        return (0, 0.1), np.arange(0, 0.101, 0.02), [f"{x:.2f}" for x in np.arange(0, 0.101, 0.02)]
    elif metric_name in ['MCC', 'AUC']:
        return (0, 1), np.arange(0, 1.01, 0.2), [f"{x:.1f}" for x in np.arange(0, 1.01, 0.2)]
    else:
        return (0, 100), np.arange(0, 101, 20), [str(x) for x in range(0, 101, 20)]

def plot_radar(metrics_dict, save_dir):
    metric_names = list(next(iter(metrics_dict.values())).keys())
    models = list(metrics_dict.keys())
    values = np.array([[m[k] for k in metric_names] for m in metrics_dict.values()])
    n = len(models)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    for i, metric in enumerate(metric_names):
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
        vals = values[:, i].tolist()
        vals.append(vals[0])
        ang = angles + angles[:1]
        ax.plot(ang, vals, linewidth=3, marker='o', markersize=7)
        ax.fill(ang, vals, alpha=0.25)
        maxv = max(vals)
        for j, a in enumerate(angles):
            ax.text(a, vals[j] + maxv * 0.08, f'{vals[j]:.3f}',
                    fontsize=16, fontweight='bold', ha='center', va='center')
        ax.set_xticks(angles)
        ax.set_xticklabels(models, fontsize=18, fontweight='bold')
        ax.tick_params(axis='x', pad=60)
        for label, ang in zip(ax.get_xticklabels(), angles):
            label.set_rotation(np.degrees(ang) - 90)
            label.set_verticalalignment('center')
        ylim, yticks, yticklabels = _radar_limits(metric)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=14, fontweight='bold')
        ax.set_title(metric, fontsize=20, fontweight='bold', pad=120)
        ax.grid(True, linewidth=1)
        plt.tight_layout()
        plt.savefig(save_dir / f'radar_{metric.replace(" ", "_").replace("/", "_")}.png', dpi=600)
        plt.close()