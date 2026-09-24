"""GraphSAGE - ein eigenes Gewicht für den Kunden, ein anderes für seine Nachbarn - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Graph-Neural-Network-Linie der "Konzepte"-Reihe (Nachfolger von gcn-demo): dasselbe Liefergebiet, dieselben Merkmale - aber jeder Kunde behält ein eigenes Gewicht für seine Merkmale, seine Nachbarn bekommen ein
anderes, und im Training darf eine Stichprobe der Nachbarn genügen.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import sg_algorithm as A
import sg_constants as C
from sg_evaluation import Settings, analyse, cost_experiment, inductive_experiment, sample_experiment, wrong_experiment
from sg_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from sg_visualization import build_cost, build_curves, build_inductive, build_map, build_sample, build_wrong, hop_distances

st.set_page_config(page_title="GraphSAGE – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


def pts(x, se=None, digits=1):
    s = f"{'+' if x >= 0 else '−'}{de(abs(100 * x), digits)}"
    return s + (f" ± {de(100 * se, digits)}" if se is not None else "")


@st.cache_data(show_spinner=False)
def _wrong(levels, seeds):
    return wrong_experiment(levels=levels, seeds=seeds)


@st.cache_data(show_spinner=False)
def _cost(noise_levels, label_levels, seeds):
    return cost_experiment(noise_levels=noise_levels, label_levels=label_levels, seeds=seeds)


@st.cache_data(show_spinner=False)
def _inductive(levels, seeds):
    return inductive_experiment(levels=levels, seeds=seeds)


@st.cache_data(show_spinner=False)
def _sample(levels, seeds, wrongs):
    return sample_experiment(levels=levels, seeds=seeds, wrongs=wrongs)


def sample_label(v):
    return "alle Nachbarn" if v == 0 else f"{v} Nachbarn"


st.title("🧭 GraphSAGE – ein Gewicht für mich, ein anderes für meine Nachbarn")
st.markdown(
    """
Im Vorgänger mittelte ein **GCN** jeden Kunden zusammen mit seinen Nachbarn und rechnete das Mittel mit **einer** Gewichtsmatrix um: der Kunde selbst zählt wie einer seiner Nachbarn. **GraphSAGE** (Hamilton/Ying/Leskovec 2017)
trennt beides: der Kunde behält **ein eigenes Gewicht** für seine Merkmale, seine Nachbarn - als Mittel - bekommen **ein anderes**. Zusätzlich darf das Training je Kunde nur eine **Stichprobe** der Nachbarn ziehen.
Die Demo misst, was das bringt und was es kostet: bei **falschen Kanten** bleibt GraphSAGE auf dem Niveau des Netzes ohne Nachbarn, bei sauberen Kanten liegt es meist **hinter** dem GCN - und "induktiv" ist es nicht mehr als das GCN.
"""
)
st.caption(
    "Viertes Stück der **Graph-Neural-Network-Linie** der \"Konzepte\"-Reihe, Nachfolger von **gcn-demo**; das Netz ist von Grund auf in numpy geschrieben, alle Daten sind erzeugt. "
    "**Bezug zu OR:** Nachbarschaftsgraphen tragen Tourenplanung und Gebietszuschnitt; große Graphen lassen sich nur über Stichproben der Nachbarschaft trainieren."
)

with st.expander("So funktioniert GraphSAGE", expanded=True):
    st.markdown(
        """
1. **Ein Gewicht für den Kunden, eines für die Nachbarn.** Eine Schicht rechnet $H' = \\mathrm{ReLU}(H\\,W_s + (P H)\\,W_n)$: $P$ mittelt über die **Nachbarn** (ohne den Kunden selbst), $W_s$ verarbeitet den Kunden, $W_n$ das Nachbarmittel.
   Gleichwertig: die Merkmale des Kunden und das Nachbarmittel aneinanderhängen und mit einer Matrix umrechnen.
2. **GCN zum Vergleich.** Das GCN rechnet $H' = \\mathrm{ReLU}(\\hat A H W)$ mit $\\hat A = D^{-1/2}(A+I)D^{-1/2}$: der Kunde geht mit denselben Gewichten in das Mittel ein wie die Nachbarn. Setzt man bei GraphSAGE $W_s = W_n$,
   bleibt ein Mittel aus dem Kunden und dem Nachbarmittel (Test in der Suite).
3. **MLP als Boden.** Mit $W_n = 0$ ist GraphSAGE genau das MLP ohne Nachbarn. Ein Netz mit eigenem Gewicht *kann* also lernen, die Nachbarn zu ignorieren.
4. **Stichprobe.** Im Training mittelt jeder Kunde je Schicht und Epoche nur über höchstens $S$ zufällig gezogene Nachbarn - das begrenzt die Rechenarbeit je Kunde. Bei der Auswertung zählen alle Nachbarn.
5. **Lernen.** Wie im Vorgänger: nur die bekannten Etiketten zählen, Adam, Gewichtszerfall, feste 200 Epochen, kein Dropout, nichts am Test abgestimmt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_nodes = st.slider("Kunden", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Zahl der Kunden im Gebiet.")
    classes = st.slider("Gebietstypen", *bounds("classes_slider"), key="classes_slider", help="Zahl der Gebietstypen (räumlich zusammenhängend, nächstes von 2 bis 4 verdeckten Zentren).")
    labels = st.slider("Bekannte Etiketten je Gebietstyp", *bounds("labels_slider"), key="labels_slider", help="Nur diese Kunden verraten dem Netz ihren Gebietstyp; alle anderen werden vorhergesagt und zur Prüfung benutzt.")
    neighbors = st.slider("Nachbarn je Kunde", *bounds("neighbors_slider"), key="neighbors_slider", help="Jeder Kunde wird mit seinen k räumlich nächsten Kunden verbunden (Kanten symmetrisch, der Grad kann darüber liegen).")
    noise = st.slider("Rauschen der Merkmale", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Streuung der Merkmale um den Mittelwert ihres Gebietstyps. Je höher, desto weniger sagt ein einzelner Kunde über seinen Typ.")
    wrong = st.slider("Anteil falscher Kanten", *bounds("wrong_slider"), key="wrong_slider", step=C.WRONG_STEP, help="Anteil der Kanten, die durch zufällige ersetzt werden (z. B. veraltete Nachbarschaftslisten); die Zahl der Kanten bleibt.")
    layers = st.slider("Schichten", *bounds("layers_slider"), key="layers_slider", help="Zahl der Schichten (für GraphSAGE, GCN und MLP gleich).")
    sample = st.selectbox("Stichprobe je Kunde und Schicht (Training)", C.SAMPLE_OPTIONS, key="sample_select", format_func=sample_label,
                          help="Wie viele Nachbarn GraphSAGE im Training je Kunde, Schicht und Epoche höchstens zieht. Die Auswertung nutzt immer alle Nachbarn; GCN und MLP kennen keine Stichprobe.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Gebiet, Merkmale, bekannte Etiketten, Stichproben und Anfangsgewichte fest.")
    st.button("🎲 Neues Gebiet generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n_nodes), "classes_slider": int(classes), "labels_slider": int(labels), "neighbors_slider": int(neighbors), "noise_slider": round(float(noise), 2), "wrong_slider": round(float(wrong), 2),
                   "layers_slider": int(layers), "sample_select": int(sample), "seed_input": int(seed)})

settings = Settings(int(n_nodes), int(classes), int(labels), int(neighbors), round(float(noise), 2), round(float(wrong), 2), int(layers), int(sample), int(seed))
with st.spinner("Trainiere GraphSAGE, GCN und MLP..."):
    a = analyse(settings)
g = a.graph
n = g.n

# --- Das Liefergebiet ---------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Das Liefergebiet und was das Netz lernt")
c1, c2 = st.columns([1, 1])
with c1:
    epoch = st.slider("Trainingsepoche", 1, C.EPOCHS, C.EPOCHS, key="epoch_slider", help="Wie weit GraphSAGE trainiert ist; die Karte zeigt seine Vorhersage nach dieser Epoche.")
with c2:
    mode = st.radio("Karte zeigt", ["Vorhersage von GraphSAGE", "Wahrer Gebietstyp"], horizontal=True, key="map_mode")
st.plotly_chart(build_map(a, epoch, "pred" if mode == "Vorhersage von GraphSAGE" else "truth"), width="stretch", key="map_chart")
pred_now = a.pred_history[epoch - 1]
unknown = ~a.train_mask
st.caption(
    f"{n} Kunden, {g.n_edges()} Kanten; {int(a.train_mask.sum())} bekannte Etiketten (schwarze Ringe). Ein Kreuz markiert eine falsche Vorhersage. Nach Epoche {epoch} liegt GraphSAGE bei "
    f"{pct(float((pred_now[unknown] == g.y[unknown]).mean()))} richtig auf den {int(unknown.sum())} unbekannten Kunden. Von den Kanten verbinden {pct(g.edge_homophily())} Kunden desselben Gebietstyps (Homophilie)."
)

st.markdown("---")

# --- Vergleich ----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 GraphSAGE gegen GCN und MLP")
m1, m2, m3, m4 = st.columns(4)
acc_s, acc_g, acc_m = a.acc("sage"), a.acc("gcn"), a.acc("mlp")
m1.metric("GraphSAGE", pct(acc_s, 1), help="Genauigkeit auf den unbekannten Kunden nach 200 Epochen (Stichprobe nur im Training).")
m2.metric("GCN", pct(acc_g, 1), delta=f"{pts(acc_s - acc_g)} Punkte GraphSAGE gegen GCN", delta_color="off", help="Ein Gewicht für den Kunden und seine Nachbarn zusammen.")
m3.metric("MLP (ohne Nachbarn)", pct(acc_m, 1), delta=f"{pts(acc_s - acc_m)} Punkte GraphSAGE gegen MLP", delta_color="off", help="Dasselbe Netz ohne Kanten.")
m4.metric("Häufigster Typ (Raten)", pct(a.majority_rate(), 1), help="Anteil des häufigsten Gebietstyps unter den unbekannten Kunden.")
st.plotly_chart(build_curves(a, epoch), width="stretch", key="curves_chart")
if acc_g < acc_m - 0.03 and acc_s > acc_g + 0.03:
    st.success(f"✅ Die falschen Kanten schaden dem GCN ({pct(acc_g, 1)} gegen {pct(acc_m, 1)} ohne Nachbarn); GraphSAGE liegt mit {pct(acc_s, 1)} ungefähr auf dem Niveau des MLP oder darüber - "
               f"es kann die Nachbarn im Zweifel ignorieren (Homophilie nur {pct(g.edge_homophily())}).")
elif acc_s > acc_g + 0.03:
    st.success(f"✅ GraphSAGE liegt vorn: {pct(acc_s, 1)} gegen {pct(acc_g, 1)} beim GCN und {pct(acc_m, 1)} ohne Nachbarn (Homophilie {pct(g.edge_homophily())}).")
elif acc_s < acc_g - 0.03:
    st.warning(f"⚠️ GraphSAGE liegt hinter dem GCN: {pct(acc_s, 1)} gegen {pct(acc_g, 1)} (MLP {pct(acc_m, 1)}). Die Nachbarn sind hier verlässlich (Homophilie {pct(g.edge_homophily())}); das eigene Gewicht "
               "bringt dann nichts, und das GCN nutzt seine Etiketten mit weniger Gewichten.")
else:
    st.info(f"Kein klarer Unterschied: GraphSAGE {pct(acc_s, 1)}, GCN {pct(acc_g, 1)}, MLP {pct(acc_m, 1)} (Homophilie {pct(g.edge_homophily())}).")

st.markdown("---")

# --- Was ein Kunde sieht ------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ein Kunde, seine Nachbarn und die Stichprobe")
default_node = int(np.argmin(np.linalg.norm(g.xy - C.AREA / 2, axis=1)))
if "node_select" not in st.session_state or st.session_state["node_select"] >= n:
    st.session_state["node_select"] = default_node
node = int(st.number_input("Kunde Nr.", 0, n - 1, key="node_select", step=1, help="Nummer eines Kunden (0 bis n-1)."))
nbrs = np.flatnonzero(g.A[node])
drawn = np.array([], dtype=int)
if settings.sample > 0:
    drawn = np.random.default_rng([settings.seed, node]).permutation(nbrs)[:settings.sample]
st.plotly_chart(build_map(a, C.EPOCHS, "pred", node=node, hops=int(layers), drawn=drawn), width="stretch", key="node_map")
own, nb_c = A.path_contributions(a.models["sage"], g.A, g.X, node)
q1, q2 = st.columns(2)
q1.metric("Eigener Beitrag (Schicht 1)", de(own, 2), help="Norm von x_i · W_s: was der Kunde selbst in die erste Schicht einbringt.")
q2.metric("Beitrag der Nachbarn (Schicht 1)", de(nb_c, 2), help="Norm von (mittlere Merkmale der Nachbarn) · W_n.")
rows = [{"Kunde": f"{node} (er selbst)", "Wahrer Typ": C.CLASS_NAMES[g.y[node]], "Gleicher Typ": "ja", "Im Beispielzug": "-"}]
for j in nbrs:
    rows.append({"Kunde": str(int(j)), "Wahrer Typ": C.CLASS_NAMES[g.y[j]], "Gleicher Typ": "ja" if g.y[j] == g.y[node] else "nein",
                 "Im Beispielzug": ("ja" if j in drawn else "nein") if settings.sample > 0 else "alle zählen"})
st.dataframe(rows, hide_index=True)
reach = int((hop_distances(g.A, node, int(layers)) >= 0).sum())
st.caption(
    f"Kunde {node} hat {len(nbrs)} Nachbarn ({int((g.y[nbrs] == g.y[node]).sum())} mit demselben Gebietstyp). "
    + (f"Im Training zieht GraphSAGE je Schicht und Epoche höchstens {settings.sample} davon (grüne Ringe zeigen einen Beispielzug; in der nächsten Epoche ist es ein anderer). " if settings.sample > 0 else "Im Training zählen alle Nachbarn. ")
    + f"Nach {int(layers)} Schicht{'en' if layers > 1 else ''} fließen Informationen von {reach} Kunden (orange hinterlegt) in seine Vorhersage ein; er selbst wird "
    f"{'richtig' if a.pred['sage'][node] == g.y[node] else 'falsch'} als {C.CLASS_NAMES[a.pred['sage'][node]]} vorhergesagt (wahr: {C.CLASS_NAMES[g.y[node]]})."
)

st.markdown("---")

# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Falsche Kanten: was bringt das eigene Gewicht?")
st.caption(f"Standardgebiet (200 Kunden, 3 Typen, 5 Etiketten je Typ, Rauschen {de(C.DEFAULT_NOISE)}); ein Anteil der Kanten wird durch zufällige ersetzt: {', '.join(pct(x) for x in C.WRONG_LEVELS)}; Mittel über {len(C.EXP_SEEDS)} feste Seeds. "
           "Dazu GraphSAGE ohne eigenes Gewicht (das Gewicht für den Kunden selbst bei 0 eingefroren). Dauer etwa 20 Sekunden.")
if st.button("Falsche Kanten durchrechnen", key="wrong_start"):
    st.session_state["wrong_on"] = True
if st.session_state.get("wrong_on"):
    rows_w = _wrong(C.WRONG_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_wrong(rows_w), width="stretch", key="wrong_chart")
    r0, rl = rows_w[0], rows_w[-1]
    best_gain = max(r["sage"] - max(r["gcn"], r["mlp"]) for r in rows_w)
    win = [r for r in rows_w if r["diff_gcn"] > 0 and r["diff_gcn"] > 1.5 * r["diff_gcn_se"]]
    st.warning(
        f"**Befund:** Ohne falsche Kanten (Homophilie {pct(r0['homophily'])}) liegt GraphSAGE mit {pct(r0['sage'], 1)} **hinter** dem GCN ({pct(r0['gcn'], 1)}; {pts(r0['diff_gcn'], r0['diff_gcn_se'])} Punkte, GCN in {r0['n_seeds'] - r0['wins_gcn']} von {r0['n_seeds']} Gebieten vorn). "
        + (f"Erst ab Homophilie {pct(win[0]['homophily'])} überholt es das GCN ({pts(win[0]['diff_gcn'], win[0]['diff_gcn_se'])} Punkte); bei vollständig zufälligen Kanten ({pct(rl['homophily'])}) sind es {pct(rl['sage'], 1)} gegen {pct(rl['gcn'], 1)} "
           f"(GraphSAGE in {rl['wins_gcn']} von {rl['n_seeds']} Gebieten vorn) und {pct(rl['mlp'], 1)} für das MLP ({pts(rl['diff_mlp'], rl['diff_mlp_se'])} Punkte). " if win else "")
        + f"Ohne das eigene Gewicht fällt GraphSAGE bei vollständig zufälligen Kanten auf {pct(rl['nonself'], 1)}: das eigene Gewicht ist es, das den Boden beim MLP-Niveau hält. "
        f"Der größte Vorsprung gegenüber dem besseren der beiden anderen Netze ({pts(best_gain)} Punkte) ist klein: GraphSAGE ist hier eine Versicherung, kein Gewinn."
    )

st.markdown("---")

# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Was kostet das eigene Gewicht bei sauberen Kanten?")
st.caption(f"Keine falschen Kanten; links wächst das Rauschen der Merkmale ({', '.join(de(x) for x in C.NOISE_LEVELS)}), rechts die Zahl bekannter Etiketten je Typ ({', '.join(str(x) for x in C.LABEL_LEVELS)}); Mittel über "
           f"{len(C.EXP_SEEDS)} feste Seeds. Dauer etwa 20 Sekunden.")
if st.button("Rauschen und Etiketten durchrechnen", key="cost_start"):
    st.session_state["cost_on"] = True
if st.session_state.get("cost_on"):
    res_c = _cost(C.NOISE_LEVELS, C.LABEL_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_cost(res_c), width="stretch", key="cost_chart")
    rn, rlab = res_c["noise"], res_c["labels"]
    parts = "; ".join(f"Rauschen {de(r['noise'])}: {pts(r['diff'], r['diff_se'])}" for r in rn)
    st.warning(
        f"**Befund:** GraphSAGE minus GCN in Punkten - {parts}. Nur bei sehr geringem Rauschen liegt GraphSAGE vorn ({rn[0]['wins']} von {rn[0]['n_seeds']} Gebieten); je mehr die einzelnen Kunden rauschen, desto weiter fällt es zurück. "
        f"Bei wenigen Etiketten ist der Rückstand am größten ({pts(rlab[0]['diff'], rlab[0]['diff_se'])} Punkte bei {rlab[0]['labels']} Etiketten je Typ, {pts(rlab[-1]['diff'], rlab[-1]['diff_se'])} bei {rlab[-1]['labels']}). "
        "Ob das an den doppelt so vielen Gewichten liegt oder daran, dass das GCN den Kunden mit seinen Nachbarn mittelt und so das Rauschen dämpft, wurde nicht getrennt gemessen."
    )

st.markdown("---")

# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Neue Kunden: ist GraphSAGE induktiv, das GCN nicht?")
st.caption(f"{C.IND_N} Kunden, nur eine zufällige Hälfte ist im Training sichtbar (Merkmale, Kanten und Etiketten); geprüft wird auf der anderen Hälfte im vollen Graphen. Vergleich mit demselben Training auf dem vollen Graphen. "
           f"{len(C.EXP_SEEDS)} feste Seeds, falsche Kanten {', '.join(pct(x) for x in C.IND_WRONG_LEVELS)}. Dauer etwa eine halbe Minute.")
if st.button("Neue Kunden durchrechnen", key="ind_start"):
    st.session_state["ind_on"] = True
if st.session_state.get("ind_on"):
    rows_i = _inductive(C.IND_WRONG_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_inductive(rows_i), width="stretch", key="ind_chart")
    parts = "; ".join(f"{pct(r['wrong'])} falsche Kanten: GraphSAGE {pct(r['sage_ind'], 1)} gegen GCN {pct(r['gcn_ind'], 1)} ({pts(r['diff_ind'], r['diff_ind_se'])} Punkte), Verlust gegenüber dem vollen Graphen "
                      f"{de(100 * r['loss_sage'], 1)} bzw. {de(100 * r['loss_gcn'], 1)} Punkte" for r in rows_i)
    st.warning(
        f"**Befund:** {parts}. Beide Netze verlieren durch den fehlenden Rest des Graphen ähnlich viel, und GraphSAGE bleibt hinter dem GCN. Ein GCN lässt sich auf neue Knoten anwenden, weil seine Gewichte an keinem Knoten hängen "
        "(nur die Normierung nutzt den Graphen, und die berechnet man neu); was GraphSAGE vom GCN unterscheidet, sind die Stichprobe (nächstes Experiment) und das eigene Gewicht (Experimente oben), nicht die Anwendung auf neue Knoten."
    )

st.markdown("---")

# --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Stichprobe: wie wenige Nachbarn genügen im Training?")
st.caption(f"Dichter Graph ({C.DENSE_K} nächste Nachbarn, Grad im Mittel etwa 23), GraphSAGE im Training mit Stichprobe {', '.join('alle' if x == 0 else str(x) for x in C.SAMPLE_LEVELS)} Nachbarn je Kunde und Schicht, Auswertung mit allen; "
           f"{len(C.EXP_SEEDS)} feste Seeds, falsche Kanten {', '.join(pct(x) for x in C.SAMPLE_WRONG_LEVELS)}. Dauer etwa eine Minute.")
if st.button("Stichprobe durchrechnen", key="sample_start"):
    st.session_state["sample_on"] = True
if st.session_state.get("sample_on"):
    rows_s = _sample(C.SAMPLE_LEVELS, C.EXP_SEEDS, C.SAMPLE_WRONG_LEVELS)
    st.plotly_chart(build_sample(rows_s, C.SAMPLE_WRONG_LEVELS), width="stretch", key="sample_chart")
    clean = [r for r in rows_s if r["wrong"] == C.SAMPLE_WRONG_LEVELS[0]]
    by = {r["sample"]: r for r in clean}
    parts = "; ".join(f"{r['sample']} {'Nachbar' if r['sample'] == 1 else 'Nachbarn'}: {pct(r['acc'], 1)} ({pts(r['diff_full'], r['diff_full_se'])} Punkte, {pct(r['share'])} der Nachrichten)" for r in sorted(clean, key=lambda r: r["sample"]) if r["sample"] > 0)
    st.warning(
        f"**Befund (ohne falsche Kanten):** alle Nachbarn {pct(by[0]['acc'], 1)}; {parts}. Die Rechenarbeit sinkt mit der Stichprobe im selben Verhältnis, die Genauigkeit erst bei kleinen Stichproben deutlich. "
        "Gemessen wird die Zahl der Nachrichten (Kanten je Epoche), nicht die Rechenzeit."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Etiketten für doppelt so viele Gewichte** | Bei wenigen Etiketten und verrauschten Merkmalen liegt GraphSAGE hinter dem GCN (Experiment oben). | – |
| **Die Nachbarn sind entweder verlässlich oder egal** | Bei falschen Kanten hält das eigene Gewicht den Boden beim MLP-Niveau; wie viel ein einzelner Nachbar wert ist, kann GraphSAGE nicht unterscheiden - alle zählen im Mittel gleich. | GAT, GATv2 (Vorgänger) |
| **Mittelwert genügt als Aggregator** | Ein Mittel kann verschiedene Nachbarschaften mit gleichem Durchschnitt nicht unterscheiden; hier nur der Mittelwert-Aggregator gemessen (kein Pooling, kein LSTM). | GIN |
| **Die Stichprobe ist erwartungstreu** | Sie schätzt das Nachbarmittel unverzerrt (Test), rauscht aber; bei kleinen Stichproben sinkt die Genauigkeit (Experiment oben). | – |
| **Lokale Nachbarschaft genügt** | Information aus weit entfernten Kunden muss durch viele Schichten wandern. | Graph Transformer |
| **Erzeugte Daten, zwölf Gebiete je Messpunkt** | Ein Vehikel mit vier Merkmalen; die Zahlen gelten für diese Größen, Standardfehler sind groß. | – |
"""
)
st.caption("Die Linie: GCN → GraphSAGE, GAT → GATv2, GIN → Graph Transformer (GIN und Graph Transformer noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Graph.** Knoten $i = 1..n$ mit Merkmalen $x_i \in \mathbb R^4$, Etikett $y_i$ (nur für $m$ Knoten bekannt), Nachbarschaftsmatrix $A \in \{0,1\}^{n \times n}$ (symmetrisch, $k$ nächste Nachbarn).

**GraphSAGE-Schicht (Mittelwert-Aggregator).** $P = D^{-1} A$ (Zeilenmittel über die Nachbarn, ohne Schleife; Knoten ohne Nachbarn: Nullzeile),
$H^{(l)} = \mathrm{ReLU}\big(H^{(l-1)} W_s^{(l)} + P\,H^{(l-1)} W_n^{(l)}\big)$, letzte Schicht ohne ReLU. Gleichwertig $W^{(l)} [h_i \,\Vert\, \tfrac1{|\mathcal N(i)|}\sum_{j \in \mathcal N(i)} h_j]$ mit $W = [W_s; W_n]$.

**Grenzfälle.** $W_n = 0$: MLP. $W_s = W_n = W$: $2\,\tfrac{I+P}{2}\,H\,W$, ein Mittel aus dem Knoten und seinem Nachbarmittel. GCN: $\hat A = \tilde D^{-1/2}(A+I)\tilde D^{-1/2}$, ein Gewicht $W$.

**Stichprobe.** Je Schicht und Epoche ersetzt man $P$ durch $\tilde P$: jeder Knoten mittelt über $\min(S, d_i)$ zufällig gezogene Nachbarn (ohne Zurücklegen); $\mathbb E[\tilde P] = P$. Die Auswertung nutzt $P$.

**Verlust und Gradienten.** $\mathcal L = -\frac1m \sum_{i \in \text{bekannt}} \log \mathrm{softmax}(Z_i)_{y_i} + \frac{\lambda}{2}\sum_l (\lVert W_s^{(l)} \rVert_F^2 + \lVert W_n^{(l)} \rVert_F^2)$. Mit $\delta^{(l)}$ als Gradient an $Z^{(l)}$:
$\partial\mathcal L/\partial W_s^{(l)} = H^{(l-1)\top}\delta^{(l)} + \lambda W_s^{(l)}$, $\partial\mathcal L/\partial W_n^{(l)} = (\tilde P H^{(l-1)})^\top\delta^{(l)} + \lambda W_n^{(l)}$,
$\delta^{(l-1)} = \big(\delta^{(l)} W_s^{(l)\top} + \tilde P^\top \delta^{(l)} W_n^{(l)\top}\big) \odot \mathbb 1[Z^{(l-1)} > 0]$ ($\tilde P$ ist nicht symmetrisch).

Implementiert in `sg_algorithm.py` (Schichten, Stichprobe, Gradienten, Adam), `sg_scenario.py` (Vehikel), `sg_evaluation.py` (Analyse, vier Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
