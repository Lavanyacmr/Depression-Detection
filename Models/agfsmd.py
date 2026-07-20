import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

class AGFSMD_GA:
    def __init__(self, pop_size=30, max_iter=100, mutation_prob=0.10, crossover_prob=0.80,
                 base_classifier=None, random_state=42):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.base_clf = base_classifier or LogisticRegression(max_iter=1000)
        self.random_state = random_state
        self.best_mask = None
        self.best_score = -1

    def _fitness(self, mask, X, y):
        if mask.sum() == 0:
            return 0.0
        X_sel = X[:, mask]
        self.base_clf.fit(X_sel, y)
        pred = self.base_clf.predict(X_sel)
        return accuracy_score(y, pred)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        n_features = X.shape[1]
        population = [np.random.choice([True, False], n_features) for _ in range(self.pop_size)]
        fitness = np.array([self._fitness(ch, X, y) for ch in population])
        best_idx = np.argmax(fitness)
        self.best_mask = population[best_idx].copy()
        self.best_score = fitness[best_idx]
        for _ in range(self.max_iter):
            new_pop = []
            for _ in range(self.pop_size // 2):
                p1 = self._select(population, fitness)
                p2 = self._select(population, fitness)
                if random.random() < self.crossover_prob:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.extend([c1, c2])
            population = new_pop[:self.pop_size]
            fitness = np.array([self._fitness(ch, X, y) for ch in population])
            cur_best = np.argmax(fitness)
            if fitness[cur_best] > self.best_score:
                self.best_score = fitness[cur_best]
                self.best_mask = population[cur_best].copy()
        return self

    def fit_predict(self, X_train, y_train, X_test):
        self._X_train = X_train
        self._y_train = y_train
        self.fit(X_train, y_train)
        X_test_sel = X_test[:, self.best_mask]
        self.base_clf.fit(X_train[:, self.best_mask], y_train)
        return self.base_clf.predict(X_test_sel)

    def _select(self, pop, fit):
        total = fit.sum()
        if total == 0:
            return pop[np.random.randint(len(pop))]
        r = np.random.rand() * total
        cum = 0
        for i, f in enumerate(fit):
            cum += f
            if cum >= r:
                return pop[i]
        return pop[-1]

    def _crossover(self, p1, p2):
        point = random.randint(1, len(p1) - 1)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2

    def _mutate(self, chrom):
        for i in range(len(chrom)):
            if random.random() < self.mutation_prob:
                chrom[i] = not chrom[i]
        return chrom