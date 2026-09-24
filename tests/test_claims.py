"""Jede Zahl aus README und PRESET_HELP als Test. Einzelläufe nur mit Strukturgrenzen (plattformrobust), Mehr-Seed-Zahlen mit großzügigen Bändern, Nachrichtenzahlen exakt (deterministisch aus dem Graphen)."""

import pytest

import sg_algorithm as A
import sg_evaluation as E
import sg_presets as P


def _preset(name):
    p = P.PRESETS[name]
    return E.analyse(E.Settings(p["n"], p["classes"], p["labels"], p["neighbors"], p["noise"], p["wrong"], p["layers"], p["sample"], p["seed"]))


def test_standard_preset_both_graph_networks_beat_the_mlp_by_a_lot():
    a = _preset("Standardfall")
    assert a.acc("sage") - a.acc("mlp") > 0.2 and a.acc("gcn") - a.acc("mlp") > 0.2 and abs(a.acc("sage") - a.acc("gcn")) < 0.1
    assert 0.88 < a.graph.edge_homophily() < 0.94


def test_eighty_percent_wrong_edges_gcn_falls_below_the_mlp_and_sage_stays_near_or_above():
    a = _preset("Viele falsche Nachbarn (80 %)")
    assert a.acc("gcn") < a.acc("mlp") and a.acc("sage") > a.acc("gcn") + 0.03 and a.acc("sage") >= a.acc("mlp") - 0.05
    assert a.graph.edge_homophily() < 0.55


def test_all_random_edges_sage_holds_the_mlp_level_gcn_below():
    a = _preset("Nur zufällige Nachbarn (100 %)")
    assert a.acc("gcn") < a.acc("mlp") - 0.05 and a.acc("sage") >= a.acc("mlp") - 0.06 and a.graph.edge_homophily() < 0.45


def test_noisy_preset_sage_far_behind_gcn():
    a = _preset("Stark verrauschte Merkmale")
    assert a.acc("gcn") - a.acc("sage") > 0.05 and a.acc("sage") > a.acc("mlp")


def test_low_noise_preset_sage_close_to_or_ahead_of_gcn():
    a = _preset("Kaum Rauschen")
    assert a.acc("sage") > a.acc("mlp") + 0.2 and abs(a.acc("sage") - a.acc("gcn")) < 0.06


def test_dense_preset_sampling_message_counts_are_exact_and_accuracy_costs():
    a = _preset("Dichter Graph, Stichprobe 2")
    assert a.models["sage"].history["messages"][0] == 800.0 == A.sample_messages(a.graph.A, 2, 2)
    full = E.analyse(E.Settings(neighbors=10, seed=11))
    assert full.models["sage"].history["messages"][0] == A.sample_messages(full.graph.A, 0, 2) == 4648.0
    assert a.acc("sage") < full.acc("sage") + 0.03 and a.acc("gcn") > a.acc("sage") + 0.03 and 11.0 < a.graph.A.sum(axis=1).mean() < 12.0


@pytest.fixture(scope="module")
def wrong_rows():
    return {r["wrong"]: r for r in E.wrong_experiment()}


def test_wrong_experiment_sage_behind_gcn_when_clean_ahead_when_random(wrong_rows):
    r = wrong_rows
    assert -0.06 < r[0.0]["diff_gcn"] < -0.005 and r[0.0]["wins_gcn"] <= 3
    assert r[0.8]["diff_gcn"] > 0.02 and r[1.0]["diff_gcn"] > 0.04 and r[1.0]["wins_gcn"] >= 7


def test_wrong_experiment_sage_holds_the_mlp_floor_and_needs_the_self_weight(wrong_rows):
    r = wrong_rows
    assert r[1.0]["diff_mlp"] > -0.06 and r[0.8]["diff_mlp"] > -0.05 and r[0.0]["diff_mlp"] > 0.15
    assert r[1.0]["sage"] - r[1.0]["nonself"] > 0.04 and r[0.8]["sage"] - r[0.8]["nonself"] > 0.06 and r[1.0]["gcn"] < r[1.0]["mlp"] - 0.05
    assert max(x["sage"] - max(x["gcn"], x["mlp"]) for x in r.values()) < 0.05
    assert r[0.0]["homophily"] > 0.88 and r[1.0]["homophily"] < 0.45


@pytest.fixture(scope="module")
def cost_rows():
    return E.cost_experiment()


def test_cost_experiment_sage_wins_only_at_low_noise_and_falls_behind_with_noise(cost_rows):
    n = {r["noise"]: r for r in cost_rows["noise"]}
    assert n[0.5]["diff"] > 0.01 and n[0.5]["wins"] >= 8 and n[3.5]["diff"] < -0.08 and n[2.5]["diff"] < -0.05 and n[3.5]["diff"] < n[1.0]["diff"]
    assert all(n[k]["sage"] > n[k]["mlp"] for k in n)


def test_cost_experiment_sage_behind_gcn_at_every_label_level(cost_rows):
    lab = {r["labels"]: r for r in cost_rows["labels"]}
    assert lab[2]["diff"] < -0.03 and all(lab[k]["diff"] < 0.0 for k in lab) and lab[20]["diff"] > lab[2]["diff"]


@pytest.fixture(scope="module")
def ind_rows():
    return {r["wrong"]: r for r in E.inductive_experiment()}


def test_inductive_experiment_gcn_is_just_as_inductive_and_ahead(ind_rows):
    r = ind_rows
    assert r[0.0]["diff_ind"] < -0.02 and r[0.4]["diff_ind"] < 0.01
    for w in r:
        assert 0.0 < r[w]["loss_sage"] < 0.12 and 0.0 < r[w]["loss_gcn"] < 0.15 and abs(r[w]["loss_sage"] - r[w]["loss_gcn"]) < 0.06
    assert r[0.4]["loss_sage"] > r[0.0]["loss_sage"]


@pytest.fixture(scope="module")
def sample_rows():
    rows = E.sample_experiment()
    return {(r["wrong"], r["sample"]): r for r in rows}


def test_sample_experiment_message_shares_and_accuracy_cost(sample_rows):
    r = sample_rows
    assert r[(0.0, 1)]["share"] == pytest.approx(0.044, abs=0.004) and r[(0.0, 2)]["share"] == pytest.approx(0.087, abs=0.005) and r[(0.0, 5)]["share"] == pytest.approx(0.219, abs=0.01)
    assert r[(0.0, 10)]["share"] == pytest.approx(0.437, abs=0.015) and r[(0.0, 0)]["messages"] == pytest.approx(9149, rel=0.02)
    assert r[(0.0, 10)]["diff_full"] > -0.06 and r[(0.0, 5)]["diff_full"] < -0.01 and r[(0.0, 2)]["diff_full"] < -0.06 and r[(0.0, 1)]["diff_full"] < -0.1
    assert r[(0.0, 1)]["acc"] < r[(0.0, 5)]["acc"] < r[(0.0, 0)]["acc"] + 0.03 and r[(0.4, 1)]["acc"] < r[(0.4, 0)]["acc"] - 0.05
