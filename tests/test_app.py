"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Kartenansichten, Kundenwahl, Stichprobe, Würfel-Knopf, Permalink-Grenzen, Extremwerte, vier Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import sg_constants as C
import sg_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_shows_metrics_and_charts():
    at = _run()
    _ok(at)
    assert len(at.metric) >= 4 and len(at.get("plotly_chart")) >= 3


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_wrong_edges_preset_shows_the_success_verdict():
    at = _run()
    next(b for b in at.button if b.key == "preset_Nur zufällige Nachbarn (100 %)").click().run()
    _ok(at)
    assert any("falschen Kanten schaden dem GCN" in s.value for s in at.success)


def test_noisy_preset_shows_graphsage_behind():
    at = _run()
    next(b for b in at.button if b.key == "preset_Stark verrauschte Merkmale").click().run()
    _ok(at)
    assert any("GraphSAGE liegt hinter dem GCN" in w.value for w in at.warning)


def test_map_views_and_epoch_slider_run():
    at = _run()
    for view in ("Vorhersage von GraphSAGE", "Wahrer Gebietstyp"):
        at.radio(key="map_mode").set_value(view).run()
        _ok(at)
    at.slider(key="epoch_slider").set_value(1).run()
    _ok(at)


def test_sample_setting_draws_neighbors_and_node_selection_survives_shrinking_n():
    at = _run(n_slider=400, node_select=399, sample_select=2, neighbors_slider=8)
    _ok(at)
    at.slider(key="n_slider").set_value(120).run()
    _ok(at)
    assert at.session_state["node_select"] < 120 and any("höchstens 2 davon" in c.value for c in at.caption)


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Gebiet generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["n"] = "9999"
    at.query_params["noise"] = "1.6"
    at.query_params["wrong"] = "0.33"
    at.query_params["sample"] = "4"
    at.query_params["layers"] = "abc"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["noise_slider"] == 1.5 and at.session_state["wrong_slider"] == 0.35
    assert at.session_state["sample_select"] == C.DEFAULT_SAMPLE and at.session_state["layers_slider"] == C.DEFAULT_LAYERS


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN, classes_slider=4, labels_slider=20), dict(n_slider=C.N_MAX, neighbors_slider=10, sample_select=1), dict(wrong_slider=1.0, layers_slider=4), dict(layers_slider=1, sample_select=8),
                                dict(noise_slider=C.NOISE_MAX), dict(labels_slider=C.LABELS_MIN, neighbors_slider=C.NEIGHBORS_MIN, sample_select=5)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_wrong_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "WRONG_LEVELS", (0.0, 1.0))
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    at = _run()
    next(b for b in at.button if b.key == "wrong_start").click().run()
    _ok(at)
    assert at.session_state["wrong_on"] and any("ohne das eigene Gewicht" in w.value.lower() or "Ohne das eigene Gewicht" in w.value for w in at.warning)


def test_cost_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "NOISE_LEVELS", (0.5, 2.0))
    monkeypatch.setattr(C, "LABEL_LEVELS", (3, 6))
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    at = _run()
    next(b for b in at.button if b.key == "cost_start").click().run()
    _ok(at)
    assert at.session_state["cost_on"] and any("GraphSAGE minus GCN in Punkten" in w.value for w in at.warning)


def test_inductive_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "IND_WRONG_LEVELS", (0.0,))
    monkeypatch.setattr(C, "IND_N", 100)
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    at = _run()
    next(b for b in at.button if b.key == "ind_start").click().run()
    _ok(at)
    assert at.session_state["ind_on"] and any("Beide Netze verlieren" in w.value for w in at.warning)


def test_sample_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "SAMPLE_LEVELS", (1, 5, 0))
    monkeypatch.setattr(C, "SAMPLE_WRONG_LEVELS", (0.0,))
    monkeypatch.setattr(C, "DENSE_K", 8)
    monkeypatch.setattr(C, "EXP_SEEDS", (0,))
    at = _run()
    next(b for b in at.button if b.key == "sample_start").click().run()
    _ok(at)
    assert at.session_state["sample_on"] and any("alle Nachbarn" in w.value and "Nachrichten" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "{pts(" not in el.value
