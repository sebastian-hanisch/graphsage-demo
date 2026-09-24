"""Plotly-Abbildungen der GraphSAGE-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sg_constants as C

PALETTE = ["#4c78a8", "#e45756", "#54a24b", "#b279a2"]
REF_COLOR = "#7f7f7f"
GOOD = "#54a24b"
WARN = "#f58518"
SAGE_COLOR = "#17becf"
GCN_COLOR = "#b279a2"
MLP_COLOR = "#4c78a8"
NONSELF_COLOR = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def hop_distances(A, source, max_hops):
    """Zahl der Kanten vom Knoten `source` bis zu jedem Knoten (breite Suche); nicht erreichbar oder weiter als max_hops: -1."""
    n = len(A)
    dist = np.full(n, -1)
    dist[source] = 0
    frontier = [source]
    for h in range(1, max_hops + 1):
        nxt = []
        for u in frontier:
            for v in np.flatnonzero(A[u]):
                if dist[v] < 0:
                    dist[v] = h
                    nxt.append(v)
        frontier = nxt
    return dist


def build_map(a, epoch, mode="pred", node=None, hops=0, drawn=None):
    """Kunden im Gebiet: Farbe = vorhergesagter (oder wahrer) Gebietstyp, x = falsch vorhergesagt, schwarzer Ring = bekanntes Etikett; optional Empfangsfeld eines Knotens und die gezogene Stichprobe seiner Nachbarn."""
    g = a.graph
    pred = a.pred_history[epoch - 1] if mode == "pred" else g.y
    fig = go.Figure()
    iu = np.array(np.nonzero(np.triu(g.A, 1)))
    xs, ys = [], []
    for i, j in zip(*iu):
        xs += [g.xy[i, 0], g.xy[j, 0], None]
        ys += [g.xy[i, 1], g.xy[j, 1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(150,150,150,0.35)", width=0.8), hoverinfo="skip", showlegend=False))
    if node is not None and hops > 0:
        d = hop_distances(g.A, node, hops)
        reach = np.flatnonzero(d >= 0)
        fig.add_trace(go.Scatter(x=g.xy[reach, 0], y=g.xy[reach, 1], mode="markers", marker=dict(size=19, color="rgba(245,133,24,0.28)"), name=f"Empfangsfeld ({hops} Schicht{'en' if hops > 1 else ''})", hoverinfo="skip"))
    for c in range(g.n_classes):
        ok = (pred == c) & (pred == g.y)
        bad = (pred == c) & (pred != g.y)
        fig.add_trace(go.Scatter(x=g.xy[ok, 0], y=g.xy[ok, 1], mode="markers", marker=dict(size=8, color=PALETTE[c], line=dict(color="white", width=0.6)), name=C.CLASS_NAMES[c],
                                 hovertemplate=f"{C.CLASS_NAMES[c]}<extra></extra>"))
        if mode == "pred" and bad.any():
            fig.add_trace(go.Scatter(x=g.xy[bad, 0], y=g.xy[bad, 1], mode="markers", marker=dict(size=10, color=PALETTE[c], symbol="x", line=dict(color=PALETTE[c], width=2)), showlegend=False,
                                     hovertemplate=f"vorhergesagt: {C.CLASS_NAMES[c]} (falsch)<extra></extra>"))
    tr = a.train_mask
    fig.add_trace(go.Scatter(x=g.xy[tr, 0], y=g.xy[tr, 1], mode="markers", marker=dict(size=14, symbol="circle-open", color="black", line=dict(width=2)), name="bekanntes Etikett"))
    if node is not None and drawn is not None and len(drawn):
        fig.add_trace(go.Scatter(x=g.xy[drawn, 0], y=g.xy[drawn, 1], mode="markers", marker=dict(size=20, symbol="circle-open", color=GOOD, line=dict(width=3)), name="gezogene Nachbarn"))
    if node is not None:
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", marker=dict(size=17, symbol="star", color=WARN, line=dict(color="black", width=1)), name="gewählter Kunde"))
    fig.update_xaxes(range=[-2, C.AREA + 2], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y")
    fig.update_yaxes(range=[-2, C.AREA + 2], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, 470).update_layout(legend=dict(orientation="h", y=-0.05), margin=dict(l=10, r=10, t=10, b=10))


def build_curves(a, epoch):
    """Links: Genauigkeit auf den unbekannten Knoten (GraphSAGE, GCN, MLP) über die Epochen; rechts: Trainingsverlust von GraphSAGE und GCN."""
    E = len(a.models["sage"].history["loss"])
    xs = list(range(1, E + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Genauigkeit auf unbekannten Kunden", "Trainingsverlust"), horizontal_spacing=0.12)
    for kind, name, color in (("sage", "GraphSAGE", SAGE_COLOR), ("gcn", "GCN", GCN_COLOR), ("mlp", "MLP (ohne Nachbarn)", MLP_COLOR)):
        fig.add_trace(go.Scatter(x=xs, y=[100 * v for v in a.models[kind].history["test_acc"]], mode="lines", name=name, line=dict(color=color, width=2.5)), row=1, col=1)
    for kind, name, color in (("sage", "GraphSAGE", SAGE_COLOR), ("gcn", "GCN", GCN_COLOR)):
        fig.add_trace(go.Scatter(x=xs, y=a.models[kind].history["loss"], mode="lines", name=name, line=dict(color=color, width=2.5), showlegend=False), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=epoch, line=dict(color=WARN, dash="dash"), row=1, col=col)
    fig.update_xaxes(title_text="Epoche")
    fig.update_yaxes(title_text="Prozent", range=[0, 102], row=1, col=1)
    fig.update_yaxes(title_text="Kreuzentropie + Zerfall", row=1, col=2)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _line(fig, xs, rows, key, name, color, dash=None, col=None):
    trace = go.Scatter(x=xs, y=[100 * r[key] for r in rows], error_y=dict(type="data", array=[100 * r[key + "_se"] for r in rows]), mode="lines+markers", name=name, line=dict(color=color, width=2.5, dash=dash))
    fig.add_trace(trace, **({"row": 1, "col": col} if col else {}))


def build_wrong(rows):
    xs = [100 * r["homophily"] for r in rows]
    fig = go.Figure()
    _line(fig, xs, rows, "sage", "GraphSAGE", SAGE_COLOR)
    _line(fig, xs, rows, "gcn", "GCN", GCN_COLOR)
    _line(fig, xs, rows, "mlp", "MLP (ohne Nachbarn)", MLP_COLOR)
    _line(fig, xs, rows, "nonself", "GraphSAGE ohne eigenes Gewicht", NONSELF_COLOR, dash="dot")
    fig.update_xaxes(title_text="Anteil der Kanten zwischen Kunden desselben Gebietstyps (%)", autorange="reversed")
    fig.update_yaxes(title_text="Genauigkeit auf unbekannten Kunden (%)", range=[30, 100])
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_cost(res):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Rauschen der Merkmale", "Bekannte Etiketten je Gebietstyp"), horizontal_spacing=0.12)
    for col, axis, rows in ((1, "noise", res["noise"]), (2, "labels", res["labels"])):
        xs = [r[axis] for r in rows]
        for kind, name, color in (("sage", "GraphSAGE", SAGE_COLOR), ("gcn", "GCN", GCN_COLOR), ("mlp", "MLP (ohne Nachbarn)", MLP_COLOR)):
            _line(fig, xs, rows, kind, name, color, col=col)
    for tr in fig.data[3:]:
        tr.showlegend = False
    fig.update_xaxes(title_text="Rauschen (Streuung der Merkmale)", row=1, col=1)
    fig.update_xaxes(title_text="Etiketten je Typ", type="log", tickvals=[r["labels"] for r in res["labels"]], ticktext=[str(r["labels"]) for r in res["labels"]], row=1, col=2)
    fig.update_yaxes(title_text="Genauigkeit auf unbekannten Kunden (%)", range=[30, 100])
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_inductive(rows):
    labels = [f"{int(100 * r['wrong'])} % falsche Kanten" for r in rows]
    fig = go.Figure()
    for key, name, color, pattern in (("sage_ind", "GraphSAGE, induktiv", SAGE_COLOR, ""), ("sage_tr", "GraphSAGE, voller Graph im Training", SAGE_COLOR, "/"),
                                      ("gcn_ind", "GCN, induktiv", GCN_COLOR, ""), ("gcn_tr", "GCN, voller Graph im Training", GCN_COLOR, "/")):
        fig.add_trace(go.Bar(x=labels, y=[100 * r[key] for r in rows], error_y=dict(type="data", array=[100 * r[key + "_se"] for r in rows]), name=name, marker=dict(color=color, pattern_shape=pattern)))
    fig.update_yaxes(title_text="Genauigkeit auf neuen Kunden (%)", range=[30, 100])
    return _base(fig, 340).update_layout(barmode="group", legend=dict(orientation="h", y=-0.35))


def build_sample(rows, wrongs):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Genauigkeit auf unbekannten Kunden", "Nachrichten je Epoche"), horizontal_spacing=0.13, specs=[[{}, {}]])
    palette = {0.0: SAGE_COLOR, 0.4: WARN}
    for w in wrongs:
        rr = [r for r in rows if r["wrong"] == w]
        rr = sorted(rr, key=lambda r: (r["sample"] == 0, r["sample"]))
        labels = ["alle" if r["sample"] == 0 else str(r["sample"]) for r in rr]
        name = f"{int(100 * w)} % falsche Kanten"
        fig.add_trace(go.Scatter(x=labels, y=[100 * r["acc"] for r in rr], error_y=dict(type="data", array=[100 * r["acc_se"] for r in rr]), mode="lines+markers", name=name, line=dict(color=palette.get(w, REF_COLOR), width=2.5)), row=1, col=1)
        if w == wrongs[0]:
            fig.add_trace(go.Bar(x=labels, y=[r["messages"] for r in rr], name="Nachrichten", marker=dict(color=REF_COLOR), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Nachbarn je Kunde und Schicht in der Stichprobe", type="category", row=1, col=1)
    fig.update_xaxes(title_text="Nachbarn je Kunde und Schicht in der Stichprobe", type="category", row=1, col=2)
    fig.update_yaxes(title_text="Prozent", range=[30, 100], row=1, col=1)
    fig.update_yaxes(title_text="Kanten je Epoche (alle Schichten)", row=1, col=2)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)
