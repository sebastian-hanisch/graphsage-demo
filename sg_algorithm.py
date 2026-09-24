"""GraphSAGE (Hamilton/Ying/Leskovec 2017) mit Mittelwert-Aggregator, von Grund auf in numpy, dazu GCN (Kipf/Welling) und MLP als Vergleich.

SAGE-Schicht:  H' = ReLU( H Ws + (P H) Wn )   mit P = Zeilenmittel über die Nachbarn (ohne den Knoten selbst) - der Knoten hat ein EIGENES Gewicht Ws, die Nachbarn ein anderes Wn.
                (gleichwertig zu W [h_i || mean_j h_j] mit W = [Ws; Wn])
GCN-Schicht:   H' = ReLU( A_hat H W )  mit A_hat = D^-1/2 (A + I) D^-1/2 - der Knoten geht mit denselben Gewichten in den Mittelwert ein wie seine Nachbarn.
MLP:           GCN mit A_hat = I.
Training von GraphSAGE mit Stichprobe: je Epoche und Schicht werden höchstens S zufällige Nachbarn je Knoten gezogen (P aus der Stichprobe); die Auswertung nutzt alle Nachbarn.
Letzte Schicht ohne ReLU, Softmax-Kreuzentropie auf den bekannten Knoten, Gewichtszerfall auf allen Matrizen, Adam, feste Epochenzahl, kein Dropout."""

from dataclasses import dataclass, field

import numpy as np

import sg_constants as C

KINDS = ("sage", "nonself", "gcn", "mlp")


def normalized_adjacency(A):
    B = A + np.eye(len(A))
    inv = 1.0 / np.sqrt(B.sum(axis=1))
    return B * inv[:, None] * inv[None, :]


def mean_matrix(A):
    """Zeilenmittel über die Nachbarn ohne den Knoten selbst; Knoten ohne Nachbarn: Nullzeile."""
    d = A.sum(axis=1, keepdims=True)
    return np.divide(A, d, out=np.zeros_like(A, dtype=float), where=d > 0)


def sample_matrix(A, S, rng):
    """Wie mean_matrix, aber jeder Knoten mittelt nur über höchstens S zufällig gezogene Nachbarn (ohne Zurücklegen)."""
    n = len(A)
    S = min(S, n)
    keys = np.where(A > 0, rng.random(A.shape), 2.0)
    idx = np.argpartition(keys, S - 1, axis=1)[:, :S]
    valid = np.take_along_axis(keys, idx, axis=1) < 2.0
    P = np.zeros((n, n))
    rows = np.repeat(np.arange(n), S)[valid.ravel()]
    P[rows, idx.ravel()[valid.ravel()]] = 1.0
    return P / np.maximum(P.sum(axis=1, keepdims=True), 1.0)


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def _glorot(rng, a, b):
    lim = np.sqrt(6.0 / (a + b))
    return rng.uniform(-lim, lim, size=(a, b))


def init_weights(kind, sizes, seed):
    """Glorot-Gleichverteilung. Rückgabe (Ws, Wn); bei gcn/mlp ist Wn None und Ws die Gewichtsmatrizen W. 'nonself': Ws bleibt Null (eingefroren)."""
    rng = np.random.default_rng([seed, 31])
    Ws, Wn = [], []
    for a, b in zip(sizes[:-1], sizes[1:]):
        Ws.append(_glorot(rng, a, b))
        if kind in ("sage", "nonself"):
            Wn.append(_glorot(rng, a, b))
    if kind == "nonself":
        Ws = [np.zeros_like(w) for w in Ws]
    return Ws, (Wn if kind in ("sage", "nonself") else None)


# --- Vorwärts- und Rückwärtsrechnung -----------------------------------------------------------------------------------------------------------------


def forward_sage(Ps, X, Ws, Wn):
    """Ps: eine Mittelwertmatrix je Schicht. Rückgabe (Zs, Hs, Logits)."""
    H = [X]
    Zs = []
    for l in range(len(Ws)):
        Z = H[-1] @ Ws[l] + (Ps[l] @ H[-1]) @ Wn[l]
        Zs.append(Z)
        H.append(np.maximum(Z, 0.0) if l < len(Ws) - 1 else Z)
    return Zs, H, H[-1]


def forward_gcn(Ahat, X, W):
    H = [X]
    Zs = []
    for l in range(len(W)):
        Z = Ahat @ H[-1] @ W[l]
        Zs.append(Z)
        H.append(np.maximum(Z, 0.0) if l < len(W) - 1 else Z)
    return Zs, H, H[-1]


def _dlogits(logits, y, train):
    P = softmax(logits)
    idx = np.flatnonzero(train)
    ce = -np.log(P[idx, y[idx]] + C.EPS).mean()
    dZ = np.zeros_like(logits)
    dZ[idx] = P[idx]
    dZ[idx, y[idx]] -= 1.0
    return ce, dZ / len(idx)


def loss_and_grads_sage(Ps, X, y, train, Ws, Wn, wd):
    Zs, H, logits = forward_sage(Ps, X, Ws, Wn)
    ce, dZ = _dlogits(logits, y, train)
    reg = 0.5 * wd * (sum(float((w ** 2).sum()) for w in Ws) + sum(float((w ** 2).sum()) for w in Wn))
    gs, gn = [None] * len(Ws), [None] * len(Ws)
    for l in range(len(Ws) - 1, -1, -1):
        gs[l] = H[l].T @ dZ + wd * Ws[l]
        gn[l] = (Ps[l] @ H[l]).T @ dZ + wd * Wn[l]
        if l > 0:
            dH = dZ @ Ws[l].T + Ps[l].T @ (dZ @ Wn[l].T)
            dZ = dH * (Zs[l - 1] > 0)
    return ce + reg, gs, gn


def loss_and_grads_gcn(Ahat, X, y, train, W, wd):
    Zs, H, logits = forward_gcn(Ahat, X, W)
    ce, dZ = _dlogits(logits, y, train)
    reg = 0.5 * wd * sum(float((w ** 2).sum()) for w in W)
    g = [None] * len(W)
    for l in range(len(W) - 1, -1, -1):
        g[l] = (Ahat @ H[l]).T @ dZ + wd * W[l]
        if l > 0:
            dZ = (Ahat @ (dZ @ W[l].T)) * (Zs[l - 1] > 0)
    return ce + reg, g


@dataclass
class Model:
    kind: str
    Ws: list
    Wn: object
    history: dict = field(default_factory=dict)


def _logits(model, A, X):
    if model.kind in ("sage", "nonself"):
        P = mean_matrix(A)
        return forward_sage([P] * len(model.Ws), X, model.Ws, model.Wn)[2]
    Ahat = normalized_adjacency(A) if model.kind == "gcn" else np.eye(len(A))
    return forward_gcn(Ahat, X, model.Ws)[2]


def predict(model, A, X):
    """Vorhersage auf einem beliebigen Graphen A (auch einem, den das Training nicht gesehen hat): die Gewichte hängen an keinem Knoten."""
    return _logits(model, A, X).argmax(axis=1)


def train(kind, A, X, y, train_mask, layers=C.DEFAULT_LAYERS, hidden=C.HIDDEN, epochs=C.EPOCHS, lr=C.LEARNING_RATE, wd=C.WEIGHT_DECAY, seed=0, sample=0, test_mask=None, eval_A=None, record_pred=False):
    """Adam-Training auf dem Graphen A. `sample` > 0: GraphSAGE mittelt je Schicht und Epoche über höchstens so viele gezogene Nachbarn. `eval_A`: Graph für die Genauigkeiten je Epoche (Standard: A);
    ein anderer Graph macht das Training induktiv. Verlauf: Verlust, Trainings- und Testgenauigkeit (Test = alle nicht bekannten Knoten, nur zur Anzeige), mittlere Nachrichten je Epoche."""
    assert kind in KINDS
    n_classes = int(y.max()) + 1
    sizes = [X.shape[1]] + [hidden] * (layers - 1) + [n_classes]
    Ws, Wn = init_weights(kind, sizes, seed)
    params = Ws + (Wn or [])
    mo = [np.zeros_like(w) for w in params]
    ve = [np.zeros_like(w) for w in params]
    rng = np.random.default_rng([seed, 5])
    is_sage = kind in ("sage", "nonself")
    Pfull = mean_matrix(A)
    Ahat = normalized_adjacency(A) if kind == "gcn" else np.eye(len(A))
    test_mask = ~train_mask if test_mask is None else test_mask
    hist = {"loss": [], "train_acc": [], "test_acc": [], "messages": []}
    preds = []
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        if is_sage:
            Ps = [sample_matrix(A, sample, rng) for _ in range(layers)] if sample > 0 else [Pfull] * layers
            loss, gs, gn = loss_and_grads_sage(Ps, X, y, train_mask, Ws, Wn, wd)
            if kind == "nonself":
                gs = [np.zeros_like(g) for g in gs]
            grads = gs + gn
            hist["messages"].append(float(sum(np.count_nonzero(P) for P in Ps)))
        else:
            loss, grads = loss_and_grads_gcn(Ahat, X, y, train_mask, Ws, wd)
            hist["messages"].append(float(layers * np.count_nonzero(A)) if kind == "gcn" else 0.0)
        for i in range(len(params)):
            mo[i] = b1 * mo[i] + (1 - b1) * grads[i]
            ve[i] = b2 * ve[i] + (1 - b2) * grads[i] ** 2
            params[i] -= lr * (mo[i] / (1 - b1 ** t)) / (np.sqrt(ve[i] / (1 - b2 ** t)) + eps)
        model = Model(kind, Ws, Wn)
        pred = predict(model, A if eval_A is None else eval_A, X)
        hist["loss"].append(float(loss))
        hist["train_acc"].append(float((pred[train_mask] == y[train_mask]).mean()))
        hist["test_acc"].append(float((pred[test_mask] == y[test_mask]).mean()))
        if record_pred:
            preds.append(pred)
    if record_pred:
        hist["pred"] = np.array(preds)
    return Model(kind, Ws, Wn, hist)


def path_contributions(model, A, X, node):
    """Norm des eigenen Beitrags x_i Ws und des Nachbarbeitrags (P x)_i Wn in der ersten Schicht (vor der Aktivierung)."""
    P = mean_matrix(A)
    own = float(np.linalg.norm(X[node] @ model.Ws[0]))
    nb = float(np.linalg.norm((P @ X)[node] @ model.Wn[0]))
    return own, nb


def sample_messages(A, S, layers):
    """Zahl der Nachrichten (Kanten) je Epoche: alle Kanten gerichtet je Schicht bzw. höchstens S je Knoten und Schicht."""
    deg = A.sum(axis=1)
    per_layer = float(deg.sum() if S <= 0 else np.minimum(deg, S).sum())
    return layers * per_layer
