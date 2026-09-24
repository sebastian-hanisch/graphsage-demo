"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import sg_constants as C
import sg_evaluation as E
import sg_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        assert (p["n"] - C.N_MIN) % C.N_STEP == 0 and p["sample"] in C.SAMPLE_OPTIONS
        for key, state_key in (("noise", "noise_slider"), ("wrong", "wrong_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-9


def test_standard_preset_equals_the_default_settings_except_for_the_seed():
    p = P.PRESETS["Standardfall"]
    assert E.Settings(p["n"], p["classes"], p["labels"], p["neighbors"], p["noise"], p["wrong"], p["layers"], p["sample"], p["seed"]) == E.Settings(seed=p["seed"])


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and set(P.STEPS) == {"n_slider", "noise_slider", "wrong_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert P.SETTING_SPECS["sample_select"].caster("3") == 3
