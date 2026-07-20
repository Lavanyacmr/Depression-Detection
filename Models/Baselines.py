from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def rf_dsm(train_texts, train_meta, train_labels, test_texts, test_meta, params):
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), stop_words='english')
    X_tr_t = vec.fit_transform(train_texts).toarray()
    X_te_t = vec.transform(test_texts).toarray()
    X_tr = np.concatenate([X_tr_t, train_meta], axis=1)
    X_te = np.concatenate([X_te_t, test_meta], axis=1)
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(X_tr, train_labels)
    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)[:, 1]
    return preds, probs

def nlp_smsd(train_texts, train_meta, train_labels, test_texts, test_meta, params):
    vec = TfidfVectorizer(max_features=params['tfidf_features'],
                          ngram_range=tuple(params['ngram_range']), stop_words='english')
    X_tr_t = vec.fit_transform(train_texts).toarray()
    X_te_t = vec.transform(test_texts).toarray()
    X_tr = np.concatenate([X_tr_t, train_meta], axis=1)
    X_te = np.concatenate([X_te_t, test_meta], axis=1)
    clf = LogisticRegression(C=params['C'], max_iter=params['max_iter'], random_state=42)
    clf.fit(X_tr, train_labels)
    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)[:, 1]
    return preds, probs

def asknn(train_texts, train_meta, train_labels, test_texts, test_meta, params):
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), stop_words='english')
    X_tr_t = vec.fit_transform(train_texts).toarray()
    X_te_t = vec.transform(test_texts).toarray()
    X_tr = np.concatenate([X_tr_t, train_meta], axis=1)
    X_te = np.concatenate([X_te_t, test_meta], axis=1)
    clf = KNeighborsClassifier(n_neighbors=params['n_neighbors'],
                               weights=params['weight_function'],
                               metric=params['distance_metric'],
                               leaf_size=params['leaf_size'])
    clf.fit(X_tr, train_labels)
    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)[:, 1]
    return preds, probs

def eec(train_texts, train_meta, train_labels, test_texts, test_meta, params):
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), stop_words='english')
    X_tr_t = vec.fit_transform(train_texts).toarray()
    X_te_t = vec.transform(test_texts).toarray()
    X_tr = np.concatenate([X_tr_t, train_meta], axis=1)
    X_te = np.concatenate([X_te_t, test_meta], axis=1)
    clf = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('gb', GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            random_state=42
        ))
    ], voting='soft')
    clf.fit(X_tr, train_labels)
    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)[:, 1]
    return preds, probs

def agf_smd_ga_run(train_texts, train_meta, train_labels, test_texts, test_meta, params):
    from models.agf_smd_ga import AGFSMD_GA
    vec = TfidfVectorizer(max_features=params['tfidf_features'],
                          ngram_range=tuple(params['ngram_range']), stop_words='english')
    X_tr_t = vec.fit_transform(train_texts).toarray()
    X_te_t = vec.transform(test_texts).toarray()
    X_tr = np.concatenate([X_tr_t, train_meta], axis=1)
    X_te = np.concatenate([X_te_t, test_meta], axis=1)
    ga = AGFSMD_GA(pop_size=params['population_size'], max_iter=params['max_iterations'],
                   mutation_prob=params['mutation_prob'], crossover_prob=params['crossover_prob'])
    preds = ga.fit_predict(X_tr, train_labels, X_te)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr[:, ga.best_mask], train_labels)
    probs = clf.predict_proba(X_te[:, ga.best_mask])[:, 1]
    return preds, probs