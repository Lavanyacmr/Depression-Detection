import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from evaluation.metrics import compute_metrics
from training.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score as sk_acc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, optimizer, scheduler, criterion, grad_clip=1.0, scaler=None, use_meta=True):
    model.train()
    total_loss = 0
    preds, labels = [], []
    for batch in loader:
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        y = batch['label'].to(DEVICE)
        if use_meta and 'metadata' in batch:
            meta = batch['metadata'].to(DEVICE)
            if 'metadata' in model.forward.__code__.co_varnames:
                out = model(ids, mask, meta)
            else:
                out = model(ids, mask)
        else:
            out = model(ids, mask)
        loss = criterion(out, y)
        optimizer.zero_grad()
        if scaler:
            with autocast():
                loss.backward()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        preds.extend((out > 0.5).float().cpu().numpy().flatten())
        labels.extend(y.cpu().numpy().flatten())
    return total_loss / len(loader), sk_acc(labels, preds)

@torch.no_grad()
def evaluate(model, loader, criterion, use_meta=True):
    model.eval()
    total_loss = 0
    preds, labels, probs = [], [], []
    for batch in loader:
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        y = batch['label'].to(DEVICE)
        if use_meta and 'metadata' in batch:
            meta = batch['metadata'].to(DEVICE)
            if 'metadata' in model.forward.__code__.co_varnames:
                out = model(ids, mask, meta)
            else:
                out = model(ids, mask)
        else:
            out = model(ids, mask)
        loss = criterion(out, y)
        total_loss += loss.item()
        preds.extend((out > 0.5).float().cpu().numpy().flatten())
        labels.extend(y.cpu().numpy().flatten())
        probs.extend(out.cpu().numpy().flatten())
    metrics = compute_metrics(np.array(labels), np.array(preds), np.array(probs))
    metrics['loss'] = total_loss / len(loader)
    return metrics, np.array(labels), np.array(preds), np.array(probs)

def train_model(model, train_loader, val_loader, test_loader, config, use_meta=True, model_name='model'):
    model = model.to(DEVICE)
    epochs = config['epochs']
    lr = config['learning_rate']
    wd = config.get('weight_decay', 0.01)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = len(train_loader) * epochs
    warmup = int(total_steps * config.get('warmup_ratio', 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    criterion = nn.BCELoss()
    es = EarlyStopping(patience=config.get('early_stopping_patience', 2))
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    scaler = GradScaler() if config.get('use_amp', False) else None

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion,
                                      config.get('grad_clip', 1.0), scaler, use_meta)
        val_met, _, _, _ = evaluate(model, val_loader, criterion, use_meta)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_met['loss'])
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_met['accuracy'])
        es(val_met['accuracy'])
        if val_met['accuracy'] >= es.best_score:
            best_state = deepcopy(model.state_dict())
        print(f"{model_name} Epoch {epoch+1}: tr_loss={tr_loss:.4f}, val_acc={val_met['accuracy']:.4f}")
        if es.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_state)
    test_met, yt, yp, yprob = evaluate(model, test_loader, criterion, use_meta)
    return test_met, yt, yp, yprob, history