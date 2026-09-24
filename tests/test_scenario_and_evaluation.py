"""Vehikel (Reproduzierbarkeit, Homophilie, falsche Kanten, Aufteilung) und Auswertung (Analyse, vier Experimente) - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import sg_constants as C
import sg_evaluation as E
import sg_scenario as S


def test_generate_is_reproducible_and_shaped():
    a, b = S.generate(100, 3, 5, 1.5, 0.0, 4), S.generate(100, 3, 5, 1.5, 0.0, 4)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.A, b.A) and np.array_equal(a.y, b.y)
    assert a.X.shape == (100, C.N_FEATURES) and a.A.shape == (100, 100) and set(np.unique(a.y)) == {0, 1, 2}
    assert not np.array_equal(a.X, S.generate(100, 3, 5, 1.5, 0.0, 5).X)


def test_adjacency_is_symmetric_without_loops_and_each_node_has_at_least_k_neighbors():
    g = S.generate(150, 3, 6, 1.0, 0.0, 2)
    assert np.array_equal(g.A, g.A.T) and np.all(np.diag(g.A) == 0) and set(np.unique(g.A)) <= {0.0, 1.0}
    assert g.A.sum(axis=1).min() >= 6


def test_neighbors_are_spatially_closest():
    g = S.generate(60, 3, 4, 1.0, 0.0, 6)
    d = np.linalg.norm(g.xy[:, None] - g.xy[None], axis=2) + np.eye(60) * 1e9
    for i in range(60):
        nearest = set(np.argsort(d[i])[:4])
        assert nearest <= set(np.flatnonzero(g.A[i]))


def test_zones_are_spatially_coherent_so_neighbors_share_the_type():
    g = S.generate(200, 3, 5, 1.0, 0.0, 3)
    assert g.edge_homophily() > 0.85


def test_rewire_keeps_the_edge_count_and_lowers_the_homophily_monotonically():
    g = S.generate(200, 3, 5, 1.0, 0.0, 3)
    homs, counts = [], []
    for w in (0.0, 0.3, 0.6, 1.0):
        g2 = S.generate(200, 3, 5, 1.0, w, 3)
        homs.append(g2.edge_homophily())
        counts.append(g2.n_edges())
        assert np.array_equal(g2.A, g2.A.T) and np.all(np.diag(g2.A) == 0)
    assert counts == [g.n_edges()] * 4 and homs == sorted(homs, reverse=True) and homs[-1] < 0.45          # zufällige Kanten: etwa Summe der Klassenanteile zum Quadrat (~ 1/3 bis 0,4)


def test_rewire_replaces_exactly_the_requested_share_of_edges():
    rng = np.random.default_rng(1)
    g = S.generate(120, 3, 5, 1.0, 0.0, 2)
    B = S.rewire(g.A, 0.5, rng)
    kept = int((np.triu(g.A, 1) * np.triu(B, 1)).sum())
    assert kept >= round(0.5 * g.n_edges()) - 1 and int(np.triu(B, 1).sum()) == g.n_edges()


def test_split_gives_exactly_per_class_known_labels():
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    tr = S.split(y, 5, 1)
    assert tr.sum() == 15 and all(tr[y == c].sum() == 5 for c in range(3))
    assert np.array_equal(tr, S.split(y, 5, 1)) and not np.array_equal(tr, S.split(y, 5, 2))


def test_split_leaves_unknown_customers_when_labels_exceed_the_class_size():
    y = np.array([0] * 8 + [1] * 8 + [2] * 8)
    tr = S.split(y, 20, 1)
    assert all(tr[y == c].sum() == 6 for c in range(3)) and (~tr).sum() == 6


def test_inductive_split_and_induced_graph():
    g = S.generate(100, 3, 5, 1.0, 0.0, 1)
    tr, seen = S.split_inductive(g.y, 5, 0.5, 1)
    assert seen.sum() == 50 and not (tr & ~seen).any() and all(tr[g.y == c].sum() == 5 for c in range(3))
    B = S.induced(g.A, seen)
    assert B[~seen].sum() == 0 and B[:, ~seen].sum() == 0 and np.array_equal(B[np.ix_(seen, seen)], g.A[np.ix_(seen, seen)])
    assert np.array_equal(tr, S.split_inductive(g.y, 5, 0.5, 1)[0])


def test_analyse_is_cached_and_consistent():
    s = E.Settings(n=80, labels=5, seed=2)
    a = E.analyse(s)
    assert a is E.analyse(s) and set(a.models) == {"sage", "gcn", "mlp"} and a.pred_history.shape == (C.EPOCHS, 80)
    assert a.acc("sage") == pytest.approx(a.models["sage"].history["test_acc"][-1]) and 0 < a.majority_rate() <= 1


def test_wrong_experiment_rows_are_consistent(monkeypatch):
    monkeypatch.setattr(C, "EPOCHS", 30)
    rows = E.wrong_experiment(levels=(0.0, 1.0), seeds=(0, 1), base=E.Settings(n=80))
    assert rows[0]["homophily"] > rows[1]["homophily"] and all(0 <= r["wins_gcn"] <= 2 for r in rows)
    assert all(r["diff_gcn"] == pytest.approx(r["sage"] - r["gcn"]) and r["diff_mlp"] == pytest.approx(r["sage"] - r["mlp"]) for r in rows)


def test_cost_experiment_rows_are_consistent(monkeypatch):
    monkeypatch.setattr(C, "EPOCHS", 30)
    res = E.cost_experiment(noise_levels=(0.5, 2.0), label_levels=(3, 6), seeds=(0, 1), base=E.Settings(n=80))
    assert [r["noise"] for r in res["noise"]] == [0.5, 2.0] and [r["labels"] for r in res["labels"]] == [3, 6]
    assert all(r["diff"] == pytest.approx(r["sage"] - r["gcn"]) and 0 <= r["wins"] <= 2 for rows in res.values() for r in rows)


def test_inductive_experiment_rows_are_consistent(monkeypatch):
    monkeypatch.setattr(C, "EPOCHS", 30)
    rows = E.inductive_experiment(levels=(0.0, 0.4), seeds=(0, 1), base=E.Settings(labels=4), n_total=100)
    assert len(rows) == 2 and all(r["diff_ind"] == pytest.approx(r["sage_ind"] - r["gcn_ind"]) and r["loss_sage"] == pytest.approx(r["sage_tr"] - r["sage_ind"]) for r in rows)


def test_sample_experiment_counts_messages_exactly(monkeypatch):
    monkeypatch.setattr(C, "EPOCHS", 20)
    rows = E.sample_experiment(levels=(1, 3, 0), seeds=(0,), base=E.Settings(n=80), dense_k=8, wrongs=(0.0,))
    g = S.generate(80, 3, 8, C.DEFAULT_NOISE, 0.0, 0)
    by = {r["sample"]: r for r in rows}
    for smp in (1, 3, 0):
        assert by[smp]["messages"] == pytest.approx(A_msgs(g.A, smp))
    assert by[0]["share"] == pytest.approx(1.0) and by[1]["share"] < by[3]["share"] < 1.0 and by[0]["diff_full"] == 0.0


def A_msgs(A, smp):
    import sg_algorithm as alg
    return alg.sample_messages(A, smp, C.DEFAULT_LAYERS)
