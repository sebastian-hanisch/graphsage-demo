"""Kern: Mittelwertmatrix und Stichprobe, SAGE-Schicht von Hand, Gradienten gegen zentrale Differenzen, Gegenproben gegen die GCN-Rechnung (MLP-Grenzfall, geteilte Gewichte), Training, induktive Vorhersage."""

import numpy as np
import pytest

import sg_algorithm as A
import sg_scenario as S


def path_graph():
    """Pfad 0 - 1 - 2."""
    Am = np.zeros((3, 3))
    Am[0, 1] = Am[1, 0] = Am[1, 2] = Am[2, 1] = 1.0
    return Am


# --- Mittelwertmatrix und Stichprobe ------------------------------------------------------------------------------------------------------


def test_mean_matrix_by_hand_and_isolated_node():
    Am = np.zeros((4, 4))
    Am[0, 1] = Am[1, 0] = Am[1, 2] = Am[2, 1] = 1.0                         # Knoten 3 ohne Nachbarn
    P = A.mean_matrix(Am)
    assert P[1] == pytest.approx([0.5, 0, 0.5, 0]) and P[0] == pytest.approx([0, 1, 0, 0]) and P[3] == pytest.approx(np.zeros(4))


def test_sample_matrix_properties():
    g = S.generate(80, 3, 8, 1.0, 0.0, 1)
    rng = np.random.default_rng(0)
    deg = g.A.sum(axis=1)
    for s in (1, 2, 3, 5):
        P = A.sample_matrix(g.A, s, rng)
        assert np.all((P > 0).sum(axis=1) == np.minimum(deg, s))            # genau min(Grad, S) Nachbarn
        assert np.all((P > 0) <= (g.A > 0))                                  # nur echte Nachbarn
        assert P.sum(axis=1) == pytest.approx(np.ones(80))                  # Mittel: Zeilensumme 1
    assert A.sample_matrix(g.A, int(deg.max()), rng) == pytest.approx(A.mean_matrix(g.A))    # S >= Grad = alle Nachbarn


def test_sample_matrix_is_random_but_reproducible_and_unbiased():
    g = S.generate(60, 3, 8, 1.0, 0.0, 2)
    a = A.sample_matrix(g.A, 2, np.random.default_rng(1))
    b = A.sample_matrix(g.A, 2, np.random.default_rng(1))
    c = A.sample_matrix(g.A, 2, np.random.default_rng(2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    rng = np.random.default_rng(3)
    mean_P = sum(A.sample_matrix(g.A, 2, rng) for _ in range(3000)) / 3000
    assert np.abs(mean_P - A.mean_matrix(g.A)).max() < 0.06                 # die Stichprobe schätzt das Mittel über alle Nachbarn erwartungstreu


# --- Schicht von Hand -------------------------------------------------------------------------------------------------------------------------


def test_sage_layer_by_hand():
    """Pfad 0-1-2, Merkmale (1, 0), (0, 1), (2, 2); Ws = I, Wn = 2 I (eine Schicht = Logits): Knoten 1: x_1 Ws + mean(x_0, x_2) Wn = (0,1) + 2 (1,5; 1) = (3, 3); Knoten 0: (1,0) + 2 (0,1) = (1, 2); Knoten 2: (2,2) + 2 (0,1) = (2, 4)."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    P = A.mean_matrix(path_graph())
    _, _, logits = A.forward_sage([P], X, [np.eye(2)], [2 * np.eye(2)])
    assert logits[1] == pytest.approx([3, 3]) and logits[0] == pytest.approx([1, 2]) and logits[2] == pytest.approx([2, 4])


def test_sage_layer_equals_concatenation_form():
    """Gleichwertigkeit mit W [h_i || mean_j h_j]: Stapelmatrix W = [Ws; Wn]."""
    rng = np.random.default_rng(0)
    g = S.generate(30, 3, 4, 1.0, 0.0, 1)
    P = A.mean_matrix(g.A)
    Ws, Wn = rng.normal(size=(4, 5)), rng.normal(size=(4, 5))
    Z = A.forward_sage([P], g.X, [Ws], [Wn])[2]
    assert Z == pytest.approx(np.hstack([g.X, P @ g.X]) @ np.vstack([Ws, Wn]))


def test_gcn_is_sage_with_shared_weights_on_the_self_loop_mean():
    """Mit Ws = Wn = W ist (I + P) H W = 2 * ((I + P)/2) H W: ein Mittel aus dem Knoten selbst (Gewicht 1/2) und dem Nachbarmittel."""
    rng = np.random.default_rng(1)
    g = S.generate(30, 3, 4, 1.0, 0.0, 1)
    P = A.mean_matrix(g.A)
    W = rng.normal(size=(4, 5))
    Z = A.forward_sage([P], g.X, [W], [W])[2]
    assert Z == pytest.approx(2 * ((np.eye(30) + P) / 2) @ g.X @ W)


# --- Gradienten und Grenzfälle ---------------------------------------------------------------------------------------------------------------


def _numeric(loss_fn, params, idxs, h=1e-6):
    out = []
    for arr, idx in idxs:
        old = arr[idx]
        arr[idx] = old + h
        up = loss_fn()
        arr[idx] = old - h
        dn = loss_fn()
        arr[idx] = old
        out.append((up - dn) / (2 * h))
    return out


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_sage_gradients_agree_with_central_differences_also_with_a_sampled_matrix(layers):
    g = S.generate(40, 3, 5, 1.0, 0.0, 3)
    tr = S.split(g.y, 4, 3)
    rng = np.random.default_rng(4)
    for sampled in (False, True):
        Ps = [A.sample_matrix(g.A, 2, rng) if sampled else A.mean_matrix(g.A) for _ in range(layers)]
        sizes = [4] + [6] * (layers - 1) + [3]
        Ws, Wn = A.init_weights("sage", sizes, 1)
        _, gs, gn = A.loss_and_grads_sage(Ps, g.X, g.y, tr, Ws, Wn, 5e-4)
        loss = lambda: A.loss_and_grads_sage(Ps, g.X, g.y, tr, Ws, Wn, 5e-4)[0]
        for W, G in ((Ws, gs), (Wn, gn)):
            for l in range(layers):
                idxs = [(W[l], tuple(rng.integers(0, s) for s in W[l].shape)) for _ in range(4)]
                num = _numeric(loss, None, idxs)
                assert [G[l][i] for _, i in idxs] == pytest.approx(num, rel=1e-4, abs=1e-8)


def test_gcn_gradients_agree_with_central_differences():
    g = S.generate(40, 3, 5, 1.0, 0.0, 3)
    tr = S.split(g.y, 4, 3)
    Ahat = A.normalized_adjacency(g.A)
    W, _ = A.init_weights("gcn", [4, 6, 3], 1)
    _, gr = A.loss_and_grads_gcn(Ahat, g.X, g.y, tr, W, 5e-4)
    loss = lambda: A.loss_and_grads_gcn(Ahat, g.X, g.y, tr, W, 5e-4)[0]
    rng = np.random.default_rng(0)
    for l in range(2):
        idxs = [(W[l], tuple(rng.integers(0, s) for s in W[l].shape)) for _ in range(4)]
        assert [gr[l][i] for _, i in idxs] == pytest.approx(_numeric(loss, None, idxs), rel=1e-4, abs=1e-8)


def test_sage_without_neighbor_weights_is_exactly_the_mlp():
    """Wn = 0: Verlust und Gradient für Ws sind die des MLP (A_hat = I, unabhängige GCN-Rechnung)."""
    g = S.generate(40, 3, 5, 1.0, 0.0, 3)
    tr = S.split(g.y, 4, 3)
    P = A.mean_matrix(g.A)
    Ws, _ = A.init_weights("sage", [4, 6, 3], 1)
    Wn0 = [np.zeros_like(w) for w in Ws]
    ls, gs, _ = A.loss_and_grads_sage([P, P], g.X, g.y, tr, Ws, Wn0, 5e-4)
    lm, gm = A.loss_and_grads_gcn(np.eye(40), g.X, g.y, tr, Ws, 5e-4)
    assert ls == pytest.approx(lm) and all(np.allclose(a, b) for a, b in zip(gs, gm))


def test_nonself_variant_keeps_the_self_weight_at_zero():
    g = S.generate(40, 3, 5, 1.0, 0.0, 3)
    tr = S.split(g.y, 4, 3)
    m = A.train("nonself", g.A, g.X, g.y, tr, epochs=20, seed=1)
    assert all(np.all(w == 0) for w in m.Ws) and any(np.abs(w).max() > 0 for w in m.Wn)


# --- Training ------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sage", "gcn", "mlp", "nonself"])
def test_training_is_reproducible_and_reduces_the_loss(kind):
    g = S.generate(80, 3, 5, 1.0, 0.0, 3)
    tr = S.split(g.y, 6, 3)
    a = A.train(kind, g.A, g.X, g.y, tr, epochs=60, seed=2)
    b = A.train(kind, g.A, g.X, g.y, tr, epochs=60, seed=2)
    assert a.history["loss"] == b.history["loss"] and a.history["loss"][-1] < 0.6 * a.history["loss"][0]
    assert float((A.predict(a, g.A, g.X)[~tr] == g.y[~tr]).mean()) == pytest.approx(a.history["test_acc"][-1])


def test_sampled_training_is_reproducible_and_counts_messages():
    g = S.generate(80, 3, 8, 1.0, 0.0, 3)
    tr = S.split(g.y, 6, 3)
    a = A.train("sage", g.A, g.X, g.y, tr, epochs=30, seed=2, sample=2)
    b = A.train("sage", g.A, g.X, g.y, tr, epochs=30, seed=2, sample=2)
    full = A.train("sage", g.A, g.X, g.y, tr, epochs=5, seed=2)
    assert a.history["loss"] == b.history["loss"]
    assert full.history["messages"][0] == 2 * g.A.sum() == A.sample_messages(g.A, 0, 2)
    assert a.history["messages"][0] == A.sample_messages(g.A, 2, 2) == 2 * np.minimum(g.A.sum(axis=1), 2).sum()
    assert a.history["messages"][0] < full.history["messages"][0]


def test_prediction_on_a_larger_unseen_graph_uses_only_shared_weights():
    """Ein Modell, das auf einem Graphen trainiert wurde, sagt auf einem anderen Graphen (anderer Knotenzahl) vorher: die Gewichte hängen an keinem Knoten."""
    g1 = S.generate(60, 3, 5, 1.0, 0.0, 1)
    g2 = S.generate(100, 3, 5, 1.0, 0.0, 2)
    for kind in ("sage", "gcn"):
        m = A.train(kind, g1.A, g1.X, g1.y, S.split(g1.y, 6, 1), epochs=20, seed=1)
        assert A.predict(m, g2.A, g2.X).shape == (100,)


def test_train_on_subgraph_ignores_features_of_unseen_nodes():
    """Beim induktiven Training haben Kunden ohne Kanten und ohne Etikett keinen Einfluss: ändert man ihre Merkmale, bleiben die Gewichte gleich."""
    g = S.generate(100, 3, 5, 1.0, 0.0, 1)
    tr, seen = S.split_inductive(g.y, 5, 0.5, 1)
    Asub = S.induced(g.A, seen)
    X2 = g.X.copy()
    X2[~seen] += 5.0
    for kind in ("sage", "gcn"):
        a = A.train(kind, Asub, g.X, g.y, tr, epochs=15, seed=1, eval_A=g.A)
        b = A.train(kind, Asub, X2, g.y, tr, epochs=15, seed=1, eval_A=g.A)
        assert all(np.allclose(p, q) for p, q in zip(a.Ws, b.Ws))


def test_path_contributions():
    g = S.generate(60, 3, 5, 1.0, 0.0, 1)
    m = A.train("sage", g.A, g.X, g.y, S.split(g.y, 5, 1), epochs=10, seed=1)
    own, nb = A.path_contributions(m, g.A, g.X, 3)
    P = A.mean_matrix(g.A)
    assert own == pytest.approx(np.linalg.norm(g.X[3] @ m.Ws[0])) and nb == pytest.approx(np.linalg.norm((P @ g.X)[3] @ m.Wn[0]))
