import numpy as np
from scipy import stats
import pandas as pd

def mcnemar_test(y_true, pred_a, pred_b):
    a_c = (pred_a == y_true)
    b_c = (pred_b == y_true)
    b01 = np.sum(a_c & ~b_c)
    b10 = np.sum(~a_c & b_c)
    n = b01 + b10
    if n == 0:
        return 1.0, 1.0
    p_exact = min(1.0, 2 * stats.binom.cdf(min(b01, b10), n, 0.5))
    chi2 = (abs(b01 - b10) - 1)**2 / n if n > 0 else 0
    p_chi = 1 - stats.chi2.cdf(chi2, 1)
    return p_exact, p_chi

def paired_ttest(fold_accs_a, fold_accs_b):
    t_stat, p_val = stats.ttest_rel(np.array(fold_accs_a), np.array(fold_accs_b))
    return t_stat, p_val

def generate_foldwise_statistical_table(cv_fold_accs, tdd_key, output_path=None):

    models = list(cv_fold_accs.keys())
    n_folds = len(next(iter(cv_fold_accs.values())))
    fold_labels = [f'Fold {i+1}' for i in range(n_folds)]

    rows = []
    for model in models:
        row = {}
        for i, lbl in enumerate(fold_labels):
            row[lbl] = cv_fold_accs[model][i]
        row['Mean Accuracy (%)'] = np.mean(cv_fold_accs[model])
        row['Standard Deviation (SD)'] = np.std(cv_fold_accs[model], ddof=0)
        rows.append(row)

    df = pd.DataFrame(rows, index=models)

    tdd_accs = cv_fold_accs[tdd_key]
    t_stats = []
    p_vals = []
    for model in models:
        if model == tdd_key:
            t_stats.append('-')
            p_vals.append('-')
        else:
            t, p = paired_ttest(cv_fold_accs[model], tdd_accs)
            t_stats.append(f"{t:.2f}")
            p_vals.append(f"{p:.2e}")
    df['Paired t-Statistic (vs TDD-Net)'] = t_stats
    df['p-Value'] = p_vals

    col_order = fold_labels + ['Mean Accuracy (%)', 'Standard Deviation (SD)',
                               'Paired t-Statistic (vs TDD-Net)', 'p-Value']
    df = df[col_order]

    if output_path:
        df.to_csv(output_path)

    return df