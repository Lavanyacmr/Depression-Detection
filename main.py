import sys, os, time, yaml, numpy as np, glob
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from preprocessing.tokenizer import load_tokenizer
from utils.data_loader import load_and_preprocess, get_texts_and_metadata
from models.tdd_net import TDDNet
from models.roberta_text_only import RoBERTaTextOnly
from models.distilbert_text_only import DistilBERTTextOnly
from models.squeezebert_text_only import SqueezeBERTTextOnly
from models.cnn_bilstm import CNNBiLSTM
from models.asym import ASYM
from models.tmfe import TMFE
from models.speecht_rag import SpeechTRAG, RetrievalDataset
from models.baselines import rf_dsm, nlp_smsd, asknn, eec, agf_smd_ga_run
from training.train import train_model
from training.cross_validation import cv_neural, cv_sklearn_baseline
from evaluation.metrics import compute_metrics
from evaluation.statistical_tests import mcnemar_test, generate_foldwise_statistical_table
from evaluation.generate_tables import create_final_comparison_table
from evaluation.generate_figures import plot_confusion_matrix, plot_roc_curves, plot_loss_acc_curves, plot_radar
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_dir = output_dir / "cv_predictions"
    cv_dir.mkdir(exist_ok=True)

    tokenizer = load_tokenizer()
    dataset, df, _ = load_and_preprocess(cfg['data_file'], tokenizer,
                                         max_len=cfg['models']['tdd_net']['max_seq_length'])
    metadata_dim = dataset.metadata.shape[1]
    labels = dataset.labels.flatten().cpu().numpy()

    train_idx, temp_idx = train_test_split(np.arange(len(labels)), test_size=0.3,
                                           stratify=labels, random_state=cfg['splits']['random_seed'])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5,
                                         stratify=labels[temp_idx], random_state=cfg['splits']['random_seed'])
    np.savez(output_dir / 'split_indices.npz', train=train_idx, val=val_idx, test=test_idx)
    train_val_idx = np.concatenate([train_idx, val_idx])

    train_val_loader = DataLoader(Subset(dataset, train_val_idx),
                                  batch_size=cfg['models']['tdd_net']['batch_size'], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx),
                            batch_size=cfg['models']['tdd_net']['batch_size'], shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx),
                             batch_size=cfg['models']['tdd_net']['batch_size'], shuffle=False)

    train_texts, train_meta, train_lbl = get_texts_and_metadata(dataset, train_val_idx)
    test_texts, test_meta, test_lbl = get_texts_and_metadata(dataset, test_idx)

    all_results = {}
    all_preds = {}
    all_histories = {}

    def run_dl(name, model_class, config_key, use_meta=True, model_kwargs={}):
        cfg_mod = cfg['models'][config_key]
        model = model_class(**model_kwargs).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        t0 = time.time()
        met, yt, yp, yprob, hist = train_model(model, train_val_loader, val_loader, test_loader,
                                               cfg_mod, use_meta=use_meta, model_name=name)
        met['time'] = time.time() - t0
        return met, yt, yp, yprob, hist

    tdd_met, tdd_yt, tdd_yp, tdd_ypb, tdd_hist = run_dl('TDD-Net', TDDNet, 'tdd_net', use_meta=True,
                                                         model_kwargs={'metadata_dim': metadata_dim})
    all_results['Proposed TDD-Net'] = tdd_met
    all_preds['Proposed TDD-Net'] = (tdd_yt, tdd_yp, tdd_ypb)
    all_histories['Proposed TDD-Net'] = tdd_hist

    rob_met, rob_yt, rob_yp, rob_ypb, rob_hist = run_dl('RoBERTa', RoBERTaTextOnly, 'roberta', use_meta=False)
    all_results['RoBERTa'] = rob_met; all_preds['RoBERTa'] = (rob_yt, rob_yp, rob_ypb); all_histories['RoBERTa'] = rob_hist

    dist_met, dist_yt, dist_yp, dist_ypb, _ = run_dl('DistilBERT', DistilBERTTextOnly, 'distilbert', use_meta=False)
    all_results['DistilBERT'] = dist_met; all_preds['DistilBERT'] = (dist_yt, dist_yp, dist_ypb)

    sq_met, sq_yt, sq_yp, sq_ypb, _ = run_dl('SqueezeBERT', SqueezeBERTTextOnly, 'squeezebert', use_meta=False)
    all_results['SqueezeBERT'] = sq_met; all_preds['SqueezeBERT'] = (sq_yt, sq_yp, sq_ypb)

    vocab_size = tokenizer.vocab_size
    cnn_met, cnn_yt, cnn_yp, cnn_ypb, cnn_hist = run_dl('CNN-BiLSTM', CNNBiLSTM, 'cnn_bilstm', use_meta=False,
                                                         model_kwargs={'vocab_size': vocab_size})
    all_results['CNN-BiLSTM'] = cnn_met; all_preds['CNN-BiLSTM'] = (cnn_yt, cnn_yp, cnn_ypb); all_histories['CNN-BiLSTM'] = cnn_hist

    t0 = time.time()
    agf_preds, agf_probs = agf_smd_ga_run(train_texts, train_meta, train_lbl, test_texts, test_meta,
                                           cfg['models']['agf_smd_ga'])
    agf_met = compute_metrics(test_lbl, agf_preds, agf_probs)
    agf_met['time'] = time.time() - t0
    all_results['AGF-SMD'] = agf_met; all_preds['AGF-SMD'] = (test_lbl, agf_preds, agf_probs)

    t0 = time.time()
    rf_preds, rf_probs = rf_dsm(train_texts, train_meta, train_lbl, test_texts, test_meta, cfg['models']['rf_dsm'])
    rf_met = compute_metrics(test_lbl, rf_preds, rf_probs)
    rf_met['time'] = time.time() - t0
    all_results['RF-DSM'] = rf_met; all_preds['RF-DSM'] = (test_lbl, rf_preds, rf_probs)

    t0 = time.time()
    nlp_preds, nlp_probs = nlp_smsd(train_texts, train_meta, train_lbl, test_texts, test_meta, cfg['models']['nlp_smsd'])
    nlp_met = compute_metrics(test_lbl, nlp_preds, nlp_probs)
    nlp_met['time'] = time.time() - t0
    all_results['NLP-SMSD'] = nlp_met; all_preds['NLP-SMSD'] = (test_lbl, nlp_preds, nlp_probs)

    t0 = time.time()
    knn_preds, knn_probs = asknn(train_texts, train_meta, train_lbl, test_texts, test_meta, cfg['models']['asknn'])
    knn_met = compute_metrics(test_lbl, knn_preds, knn_probs)
    knn_met['time'] = time.time() - t0
    all_results['ASKNN'] = knn_met; all_preds['ASKNN'] = (test_lbl, knn_preds, knn_probs)

    t0 = time.time()
    eec_preds, eec_probs = eec(train_texts, train_meta, train_lbl, test_texts, test_meta, cfg['models']['eec'])
    eec_met = compute_metrics(test_lbl, eec_preds, eec_probs)
    eec_met['time'] = time.time() - t0
    all_results['EEC'] = eec_met; all_preds['EEC'] = (test_lbl, eec_preds, eec_probs)

    asym_met, asym_yt, asym_yp, asym_ypb, _ = run_dl('ASYM', ASYM, 'asym', use_meta=False,
                                                      model_kwargs=cfg['models']['asym'])
    all_results['ASYM'] = asym_met; all_preds['ASYM'] = (asym_yt, asym_yp, asym_ypb)

    tmfe_met, tmfe_yt, tmfe_yp, tmfe_ypb, _ = run_dl('TMFE', TMFE, 'tmfe', use_meta=False,
                                                      model_kwargs=cfg['models']['tmfe'])
    all_results['TMFE'] = tmfe_met; all_preds['TMFE'] = (tmfe_yt, tmfe_yp, tmfe_ypb)

    sent_model = SentenceTransformer('all-MiniLM-L6-v2')
    rag_train = RetrievalDataset(train_texts, train_lbl, tokenizer, sent_model, top_k=5)
    rag_val = RetrievalDataset([dataset.texts[i] for i in val_idx], labels[val_idx], tokenizer, sent_model, top_k=5)
    rag_test = RetrievalDataset(test_texts, test_lbl, tokenizer, sent_model, top_k=5)
    rag_tr_loader = DataLoader(rag_train, batch_size=16, shuffle=True)
    rag_val_loader = DataLoader(rag_val, batch_size=16, shuffle=False)
    rag_te_loader = DataLoader(rag_test, batch_size=16, shuffle=False)
    speech_model = SpeechTRAG().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    speech_met, st_yt, st_yp, st_ypb, _ = train_model(speech_model, rag_tr_loader, rag_val_loader, rag_te_loader,
                                                       cfg['models']['speecht_rag'], use_meta=False, model_name='SpeechT-RAG')
    speech_met['time'] = time.time() - t0
    all_results['SpeechT-RAG'] = speech_met; all_preds['SpeechT-RAG'] = (st_yt, st_yp, st_ypb)

    print("\n=== 5‑fold cross‑validation ===")
    cv_fold_accs = {}
    tdd_cv_met = cv_neural(TDDNet, Subset(dataset, train_idx), cfg, n_splits=cfg['kfold']['n_splits'],
                            use_meta=True, model_name='TDDNet', output_dir=cv_dir, metadata_dim=metadata_dim)
    cv_fold_accs['Proposed TDD-Net'] = [m['accuracy'] for m in tdd_cv_met]

    dl_cv_list = [
        ('RoBERTa', RoBERTaTextOnly, 'roberta', False, {}),
        ('DistilBERT', DistilBERTTextOnly, 'distilbert', False, {}),
        ('SqueezeBERT', SqueezeBERTTextOnly, 'squeezebert', False, {}),
        ('CNN-BiLSTM', CNNBiLSTM, 'cnn_bilstm', False, {'vocab_size': vocab_size}),
        ('ASYM', ASYM, 'asym', False, cfg['models']['asym']),
        ('TMFE', TMFE, 'tmfe', False, cfg['models']['tmfe']),
    ]
    for name, cls, ck, um, kw in dl_cv_list:
        cv_met = cv_neural(cls, Subset(dataset, train_idx), cfg, n_splits=cfg['kfold']['n_splits'],
                          use_meta=um, model_name=name, output_dir=cv_dir, **kw)
        cv_fold_accs[name] = [m['accuracy'] for m in cv_met]

    sklearn_cv = [
        ('RF-DSM', rf_dsm, 'rf_dsm'),
        ('NLP-SMSD', nlp_smsd, 'nlp_smsd'),
        ('ASKNN', asknn, 'asknn'),
        ('EEC', eec, 'eec'),
        ('AGF-SMD', agf_smd_ga_run, 'agf_smd_ga'),
    ]
    for name, func, pk in sklearn_cv:
        cv_met = cv_sklearn_baseline(func, dataset, cfg, cfg['models'][pk], name, output_dir=cv_dir)
        cv_fold_accs[name] = [m['accuracy'] for m in cv_met]

    # Generate fold‑wise confusion matrices
    for file in sorted(glob.glob(str(cv_dir / "fold*_preds.npz"))):
        data = np.load(file)
        y_true = data['y_true']
        y_pred = data['y_pred']
        base = os.path.splitext(os.path.basename(file))[0]
        plot_confusion_matrix(y_true, y_pred, base, cv_dir / f"cm_{base}.png")

    # Produce and save the exact fold‑wise statistical table
    fold_table = generate_foldwise_statistical_table(cv_fold_accs, 'Proposed TDD-Net',
                                                     output_path=output_dir / 'foldwise_statistical_table.csv')
    print("\n" + "="*80)
    print("FOLD‑WISE ACCURACY COMPARISON")
    print("="*80)
    print(fold_table.to_string(float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))
    print("="*80)

    mcnemar_res = {}
    for name, (_, preds, _) in all_preds.items():
        if name == 'Proposed TDD-Net':
            continue
        p_ex, p_chi = mcnemar_test(tdd_yt, tdd_yp, preds)
        mcnemar_res[name] = {'p_exact': p_ex, 'p_chi2': p_chi}
    pd.DataFrame(mcnemar_res).T.to_csv(output_dir / 'mcnemar_tests.csv')

    create_final_comparison_table(all_results, output_dir)

    for name, (yt, yp, _) in all_preds.items():
        plot_confusion_matrix(yt, yp, name, output_dir / f'cm_{name}.png')

    all_y_true = [v[0] for v in all_preds.values()]
    all_y_prob = [v[2] for v in all_preds.values()]
    plot_roc_curves(all_y_true, all_y_prob, list(all_preds.keys()), output_dir / 'roc_curves.png')

    for name, hist in all_histories.items():
        plot_loss_acc_curves(hist, name, output_dir)

    radar_dict = {}
    for name, met in all_results.items():
        radar_dict[name] = {
            'Accuracy': met['accuracy'] * 100,
            'Precision': met['precision'] * 100,
            'Recall': met['recall'] * 100,
            'Specificity': met['specificity'] * 100,
            'MCC': met['mcc'],
            'NPV': met['npv'] * 100,
            'FPR': met['fpr'],
            'FNR': met['fnr'],
            'F1': met['f1'] * 100,
            'AUC': met['auc'],
            'Time (s)': met.get('time', 0)
        }
    plot_radar(radar_dict, output_dir)

    print(f"\nAll results saved to {output_dir}")

if __name__ == '__main__':
    main()