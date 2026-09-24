"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. g2_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import sg_constants as C


def _sample(value):
    v = int(float(value))
    if v not in C.SAMPLE_OPTIONS:
        raise ValueError(value)
    return v


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "classes_slider": SettingSpec("classes", int, C.DEFAULT_CLASSES, C.CLASSES_MIN, C.CLASSES_MAX),
    "labels_slider": SettingSpec("labels", int, C.DEFAULT_LABELS, C.LABELS_MIN, C.LABELS_MAX),
    "neighbors_slider": SettingSpec("neighbors", int, C.DEFAULT_NEIGHBORS, C.NEIGHBORS_MIN, C.NEIGHBORS_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "wrong_slider": SettingSpec("wrong", float, C.DEFAULT_WRONG, C.WRONG_MIN, C.WRONG_MAX),
    "layers_slider": SettingSpec("layers", int, C.DEFAULT_LAYERS, C.LAYERS_MIN, C.LAYERS_MAX),
    "sample_select": SettingSpec("sample", _sample, C.DEFAULT_SAMPLE),
    "seed_input": SettingSpec("seed", int, 7, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "classes": "classes_slider", "labels": "labels_slider", "neighbors": "neighbors_slider", "noise": "noise_slider", "wrong": "wrong_slider", "layers": "layers_slider", "sample": "sample_select", "seed": "seed_input"}
STEPS = {"n_slider": C.N_STEP, "noise_slider": C.NOISE_STEP, "wrong_slider": C.WRONG_STEP}

def _p(**kw):
    base = {"n": 200, "classes": 3, "labels": 5, "neighbors": 5, "noise": 1.5, "wrong": 0.0, "layers": 2, "sample": 0, "seed": 0}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _p(),
    "Viele falsche Nachbarn (80 %)": _p(wrong=0.8, seed=8),
    "Nur zufällige Nachbarn (100 %)": _p(wrong=1.0),
    "Stark verrauschte Merkmale": _p(noise=3.0, seed=5),
    "Kaum Rauschen": _p(noise=0.5, seed=3),
    "Dichter Graph, Stichprobe 2": _p(neighbors=10, sample=2, seed=11),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall": "Seed 0, 200 Kunden, 5 Etiketten je Typ, saubere Kanten (Homophilie 90,8 %): GraphSAGE 91,9 %, GCN 94,1 %, MLP 48,1 % (Raten 50,3 %). Beide Netze mit Nachbarn schlagen das MLP deutlich; im Mittel über 12 Gebiete liegt GraphSAGE etwa 3 Punkte hinter dem GCN.",
    "Viele falsche Nachbarn (80 %)": "Seed 8, 80 % der Kanten zufällig ersetzt (Homophilie 45,5 %): GraphSAGE 64,3 %, MLP 61,1 %, GCN 55,1 %. Das GCN fällt unter das MLP, GraphSAGE bleibt darüber - es kann die Nachbarn im Zweifel ignorieren.",
    "Nur zufällige Nachbarn (100 %)": "Seed 0, alle Kanten zufällig (Homophilie 36,3 %): GraphSAGE 50,3 %, MLP 48,1 %, GCN 35,7 % (Raten 50,3 %). Ohne brauchbare Nachbarn hält GraphSAGE das Niveau des MLP; das GCN liegt darunter.",
    "Stark verrauschte Merkmale": "Seed 5, Rauschen 3,0: GraphSAGE 69,7 %, GCN 82,7 %, MLP 45,9 %. Bei stark verrauschten Merkmalen liegt GraphSAGE deutlich hinter dem GCN - der Rückstand wächst im Mittel mit dem Rauschen (Experiment unten).",
    "Kaum Rauschen": "Seed 3, Rauschen 0,5: GraphSAGE 96,8 %, GCN 94,6 %, MLP 60,0 %. Bei sehr geringem Rauschen liegt GraphSAGE im Mittel knapp vorn (+3 Punkte, in 11 von 12 Gebieten).",
    "Dichter Graph, Stichprobe 2": "Seed 11, 10 Nachbarn je Kunde (Grad im Mittel 11,6), Stichprobe 2 im Training: GraphSAGE 81,6 % mit 800 Nachrichten je Epoche, gegen 85,4 % mit allen Nachbarn (4.648 Nachrichten); GCN 90,8 %, MLP 60,0 %. Die Stichprobe spart viel Rechenarbeit und kostet Genauigkeit.",
}
