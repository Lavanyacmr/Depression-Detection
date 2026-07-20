import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from evaluation.metrics import compute_metrics
from utils.data_loader import get_texts_and_metadata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cv_neural(model_class, dataset, config, n_splits=5, use_meta=True, model_name='model',
              output_dir=None, **model_kwargs):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config['kfold']['random_seed'])
    labels = dataset.labels.flatten().cpu().numpy()
    fold_metrics = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"  {model_name} Fold {fold+1}")
        tr_sub = Subset(dataset, tr_idx)
        val_sub = Subset(dataset, val_idx)
        tr_loader = DataLoader(tr_sub, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=config['batch_size'], shuffle=False)
        model = model_class(**model_kwargs).to(DEVICE)
        from training.train import train_model
        met, yt, yp, yprob, _ = train_model(model, tr_loader, val_loader, val_loader, config,
                                            use_meta=use_meta, model_name=f"{model_name}_f{fold+1}")
        fold_metrics.append(met)
        if output_dir:
            np.savez(output_dir / f"fold{fold+1}_{model_name}_preds.npz",
                     y_true=yt, y_pred=yp, y_prob=yprob)
    return fold_metrics

def cv_sklearn_baseline(func, dataset, config, params, baseline_name, output_dir=None):
    skf = StratifiedKFold(n_splits=config['kfold']['n_splits'],
                          shuffle=True, random_state=config['kfold']['random_seed'])
    labels = dataset.labels.flatten().cpu().numpy()
    fold_metrics = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"  {baseline_name} Fold {fold+1}")
        train_texts, train_meta, train_lbl = get_texts_and_metadata(dataset, tr_idx)
        test_texts, test_meta, test_lbl = get_texts_and_metadata(dataset, val_idx)
        preds, probs = func(train_texts, train_meta, train_lbl, test_texts, test_meta, params)
        met = compute_metrics(test_lbl, preds, probs)
        met['fold'] = fold
        fold_metrics.append(met)
        if output_dir:
            np.savez(output_dir / f"fold{fold+1}_{baseline_name}_preds.npz",
                     y_true=test_lbl, y_pred=preds, y_prob=probs)
    return fold_metrics