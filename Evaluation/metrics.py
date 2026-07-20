import numpy as np
import math
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, log_loss

def compute_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    total = tn + tp + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den > 0 else 0
    res = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'specificity': spec,
        'npv': npv, 'fpr': fpr, 'fnr': fnr, 'f1': f1, 'mcc': mcc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    if y_prob is not None:
        try:
            res['auc'] = roc_auc_score(y_true, y_prob)
        except:
            res['auc'] = 0.5
        res['brier'] = brier_score_loss(y_true, y_prob)
        res['log_loss'] = log_loss(y_true, y_prob)
    return res