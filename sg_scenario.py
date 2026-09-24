"""Vehikel D "Liefergebiete": n Kunden in einem 100 x 100 km großen Gebiet; das Gebiet zerfällt in räumlich zusammenhängende Gebietstypen (nächstes von K verdeckten Zentren). Jeder Kunde hat vier Merkmale,
deren Mittelwerte vom Gebietstyp abhängen und die stark verrauscht sind. Der Graph verbindet räumliche Nachbarn (k nächste, symmetrisch); ein Anteil der Kanten kann durch zufällige ersetzt werden
("falsche Kanten", z. B. veraltete Nachbarschaftslisten)."""

from dataclasses import dataclass

import numpy as np

import sg_constants as C


@dataclass(frozen=True)
class Graph:
    xy: np.ndarray          # (n, 2)
    X: np.ndarray           # (n, F) Merkmale
    y: np.ndarray           # (n,) Gebietstyp
    A: np.ndarray           # (n, n) symmetrische 0/1-Nachbarschaft ohne Schleifen
    n_classes: int
    seed: int

    @property
    def n(self):
        return len(self.y)

    def edge_homophily(self):
        """Anteil der Kanten zwischen Kunden desselben Gebietstyps."""
        i, j = np.nonzero(np.triu(self.A, 1))
        return float(np.mean(self.y[i] == self.y[j])) if len(i) else 1.0

    def n_edges(self):
        return int(np.triu(self.A, 1).sum())


def _centers(rng, k):
    """K Zentren mit Mindestabstand (Ablehnungsverfahren), damit alle Gebietstypen vorkommen."""
    for _ in range(1000):
        c = rng.random((k, 2)) * C.AREA
        d = np.linalg.norm(c[:, None] - c[None], axis=2) + np.eye(k) * 1e9
        if d.min() >= 0.28 * C.AREA:
            return c
    return c


def knn_adjacency(xy, k):
    n = len(xy)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2) + np.eye(n) * 1e9
    idx = np.argsort(d, axis=1)[:, :k]
    A = np.zeros((n, n))
    A[np.repeat(np.arange(n), k), idx.ravel()] = 1.0
    return np.maximum(A, A.T)


def rewire(A, fraction, rng):
    """Ersetzt einen Anteil der Kanten durch gleich viele zufällige (keine Schleifen, keine Doppelten): die Kantenzahl bleibt, die Homophilie sinkt."""
    if fraction <= 0:
        return A.copy()
    n = len(A)
    iu = np.array(np.nonzero(np.triu(A, 1))).T
    m = len(iu)
    n_swap = int(round(fraction * m))
    keep = np.ones(m, dtype=bool)
    keep[rng.permutation(m)[:n_swap]] = False
    B = np.zeros_like(A)
    B[iu[keep, 0], iu[keep, 1]] = 1.0
    B = np.maximum(B, B.T)
    added = 0
    while added < n_swap:
        i, j = rng.integers(0, n, 2)
        if i != j and B[i, j] == 0:
            B[i, j] = B[j, i] = 1.0
            added += 1
    return B


def generate(n=C.DEFAULT_N, n_classes=C.DEFAULT_CLASSES, k=C.DEFAULT_NEIGHBORS, noise=C.DEFAULT_NOISE, wrong=C.DEFAULT_WRONG, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * C.AREA
    centers = _centers(rng, n_classes)
    y = np.argmin(np.linalg.norm(xy[:, None] - centers[None], axis=2), axis=1)
    means = rng.normal(size=(n_classes, C.N_FEATURES))
    means /= np.linalg.norm(means, axis=1, keepdims=True)                 # Einheitsabstand vom Ursprung
    X = means[y] + noise * rng.normal(size=(n, C.N_FEATURES)) * 0.5
    A = rewire(knn_adjacency(xy, k), wrong, np.random.default_rng([seed, 99]))
    return Graph(xy, X, y, A, n_classes, int(seed))


def split(y, per_class, seed):
    """Je Gebietstyp `per_class` bekannte Etiketten (Training); alle übrigen Kunden sind unbekannt und dienen der Prüfung."""
    rng = np.random.default_rng([seed, 7])
    train = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        train[rng.permutation(idx)[:min(per_class, max(len(idx) - 2, 1))]] = True          # mindestens zwei unbekannte Kunden je Typ bleiben zur Prüfung übrig
    return train


def split_inductive(y, per_class, frac_seen, seed):
    """Induktive Aufteilung: ein zufälliger Anteil `frac_seen` der Kunden ist im Training sichtbar (mit Merkmalen und Kanten); die bekannten Etiketten liegen nur unter ihnen. Rückgabe (train_mask, seen_mask); die
    nicht sichtbaren Kunden sind die Prüfmenge."""
    rng = np.random.default_rng([seed, 3])
    n = len(y)
    seen = np.zeros(n, dtype=bool)
    seen[rng.permutation(n)[:int(round(frac_seen * n))]] = True
    train = np.zeros(n, dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero((y == c) & seen)
        train[rng.permutation(idx)[:min(per_class, len(idx))]] = True
    return train, seen


def induced(A, keep):
    """Adjazenz nur zwischen den Knoten in `keep`; alle anderen Knoten bleiben ohne Kanten."""
    B = np.zeros_like(A)
    idx = np.flatnonzero(keep)
    B[np.ix_(idx, idx)] = A[np.ix_(idx, idx)]
    return B
