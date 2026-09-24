"""Auswertung: GraphSAGE gegen GCN und MLP auf einem Liefergebiets-Graphen und vier Experimente (falsche Kanten, Preis des eigenen Gewichts, neue Kunden ohne Training, Stichprobe der Nachbarn)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import sg_algorithm as A
import sg_constants as C
import sg_scenario as S


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    classes: int = C.DEFAULT_CLASSES
    labels: int = C.DEFAULT_LABELS
    neighbors: int = C.DEFAULT_NEIGHBORS
    noise: float = C.DEFAULT_NOISE
    wrong: float = C.DEFAULT_WRONG
    layers: int = C.DEFAULT_LAYERS
    sample: int = C.DEFAULT_SAMPLE
    seed: int = 7


@dataclass
class Analysis:
    settings: Settings
    graph: S.Graph
    train_mask: np.ndarray
    models: dict                    # sage, gcn, mlp
    pred: dict
    pred_history: np.ndarray        # GraphSAGE-Vorhersage je Epoche

    def acc(self, kind):
        m = ~self.train_mask
        return float((self.pred[kind][m] == self.graph.y[m]).mean())

    def majority_rate(self):
        m = ~self.train_mask
        return float(np.bincount(self.graph.y[m]).max() / m.sum())


def _generate(s):
    g = S.generate(s.n, s.classes, s.neighbors, s.noise, s.wrong, s.seed)
    return g, S.split(g.y, s.labels, s.seed)


@lru_cache(maxsize=64)
def analyse(settings):
    g, tr = _generate(settings)
    models = {
        "sage": A.train("sage", g.A, g.X, g.y, tr, layers=settings.layers, seed=settings.seed, sample=settings.sample, record_pred=True),
        "gcn": A.train("gcn", g.A, g.X, g.y, tr, layers=settings.layers, seed=settings.seed),
        "mlp": A.train("mlp", g.A, g.X, g.y, tr, layers=settings.layers, seed=settings.seed),
    }
    pred = {k: A.predict(m, g.A, g.X) for k, m in models.items()}
    return Analysis(settings, g, tr, models, pred, models["sage"].history["pred"])


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def _replace(base, **kw):
    d = dict(base.__dict__)
    d.update(kw)
    return Settings(**d)


# --- Experiment 1: falsche Kanten -------------------------------------------------------------------------------------------------------------


def wrong_experiment(levels=None, seeds=None, base=None):
    """GraphSAGE, GCN, MLP und 'nur Nachbarn' (GraphSAGE mit eingefrorenem Null-Gewicht für den Knoten selbst) bei wachsendem Anteil falscher Kanten."""
    levels = C.WRONG_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for w in levels:
        res = {k: [] for k in ("sage", "gcn", "mlp", "nonself")}
        hom = []
        for s in seeds:
            st = _replace(base, wrong=w, sample=0, seed=s)
            g, tr = _generate(st)
            hom.append(g.edge_homophily())
            for k in res:
                m = A.train(k, g.A, g.X, g.y, tr, layers=st.layers, seed=s)
                res[k].append(m.history["test_acc"][-1])
        row = {"wrong": w, "homophily": float(np.mean(hom)), "n_seeds": len(seeds)}
        for k, v in res.items():
            row[k], row[k + "_se"] = _mean_se(v)
        row["diff_gcn"], row["diff_gcn_se"] = _mean_se(np.array(res["sage"]) - np.array(res["gcn"]))
        row["diff_mlp"], row["diff_mlp_se"] = _mean_se(np.array(res["sage"]) - np.array(res["mlp"]))
        row["wins_gcn"] = int(np.sum(np.array(res["sage"]) > np.array(res["gcn"])))
        rows.append(row)
    return rows


# --- Experiment 2: Preis des eigenen Gewichts (Rauschen, Etiketten) ----------------------------------------------------------------------------


def cost_experiment(noise_levels=None, label_levels=None, seeds=None, base=None):
    """Ohne falsche Kanten: GraphSAGE minus GCN bei wachsendem Rauschen der Merkmale und bei wachsender Zahl bekannter Etiketten."""
    noise_levels = C.NOISE_LEVELS if noise_levels is None else noise_levels
    label_levels = C.LABEL_LEVELS if label_levels is None else label_levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base

    def one(axis, value):
        res = {k: [] for k in ("sage", "gcn", "mlp")}
        for s in seeds:
            st = _replace(base, sample=0, seed=s, **{axis: value})
            g, tr = _generate(st)
            for k in res:
                res[k].append(A.train(k, g.A, g.X, g.y, tr, layers=st.layers, seed=s).history["test_acc"][-1])
        row = {axis: value, "n_seeds": len(seeds)}
        for k, v in res.items():
            row[k], row[k + "_se"] = _mean_se(v)
        d = np.array(res["sage"]) - np.array(res["gcn"])
        row["diff"], row["diff_se"] = _mean_se(d)
        row["wins"] = int(np.sum(d > 0))
        return row

    return {"noise": [one("noise", v) for v in noise_levels], "labels": [one("labels", v) for v in label_levels]}


# --- Experiment 3: neue Kunden ---------------------------------------------------------------------------------------------------------------


def inductive_experiment(levels=None, seeds=None, base=None, n_total=None, frac_seen=None):
    """Training nur auf der sichtbaren Hälfte (induzierter Teilgraph); geprüft wird auf den nicht sichtbaren Kunden im vollen Graphen. Gegenprobe: dasselbe Training auf dem vollen Graphen (transduktiv)."""
    levels = C.IND_WRONG_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    n_total = C.IND_N if n_total is None else n_total
    frac_seen = C.IND_SEEN if frac_seen is None else frac_seen
    rows = []
    for w in levels:
        res = {k: [] for k in ("sage_ind", "gcn_ind", "sage_tr", "gcn_tr")}
        for s in seeds:
            st = _replace(base, n=n_total, wrong=w, sample=0, seed=s)
            g = S.generate(st.n, st.classes, st.neighbors, st.noise, st.wrong, s)
            tr, seen = S.split_inductive(g.y, st.labels, frac_seen, s)
            te = ~seen
            Asub = S.induced(g.A, seen)
            for kind in ("sage", "gcn"):
                m = A.train(kind, Asub, g.X, g.y, tr, layers=st.layers, seed=s, test_mask=te, eval_A=g.A)
                res[kind + "_ind"].append(m.history["test_acc"][-1])
                m = A.train(kind, g.A, g.X, g.y, tr, layers=st.layers, seed=s, test_mask=te)
                res[kind + "_tr"].append(m.history["test_acc"][-1])
        row = {"wrong": w, "n_seeds": len(seeds)}
        for k, v in res.items():
            row[k], row[k + "_se"] = _mean_se(v)
        row["diff_ind"], row["diff_ind_se"] = _mean_se(np.array(res["sage_ind"]) - np.array(res["gcn_ind"]))
        row["loss_sage"], row["loss_sage_se"] = _mean_se(np.array(res["sage_tr"]) - np.array(res["sage_ind"]))
        row["loss_gcn"], row["loss_gcn_se"] = _mean_se(np.array(res["gcn_tr"]) - np.array(res["gcn_ind"]))
        rows.append(row)
    return rows


# --- Experiment 4: Stichprobe der Nachbarn ------------------------------------------------------------------------------------------------


def sample_experiment(levels=None, seeds=None, base=None, dense_k=None, wrongs=None):
    """Dichter Graph (viele Nachbarn je Kunde): Genauigkeit und Zahl der Nachrichten je Epoche in Abhängigkeit von der Stichprobengröße S (0 = alle Nachbarn)."""
    levels = C.SAMPLE_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    dense_k = C.DENSE_K if dense_k is None else dense_k
    wrongs = C.SAMPLE_WRONG_LEVELS if wrongs is None else wrongs
    rows = []
    for w in wrongs:
        accs, msgs = {}, {}
        for smp in levels:
            accs[smp], msgs[smp] = [], []
            for s in seeds:
                st = _replace(base, neighbors=dense_k, wrong=w, sample=smp, seed=s)
                g, tr = _generate(st)
                m = A.train("sage", g.A, g.X, g.y, tr, layers=st.layers, seed=s, sample=smp)
                accs[smp].append(m.history["test_acc"][-1])
                msgs[smp].append(float(np.mean(m.history["messages"])))
        full = np.array(accs[0]) if 0 in accs else None
        full_msgs = float(np.mean(msgs[0])) if 0 in msgs else None
        for smp in levels:
            row = {"wrong": w, "sample": smp, "n_seeds": len(seeds), "messages": float(np.mean(msgs[smp]))}
            row["acc"], row["acc_se"] = _mean_se(accs[smp])
            if full is not None:
                row["share"] = row["messages"] / full_msgs
                row["diff_full"], row["diff_full_se"] = _mean_se(np.array(accs[smp]) - full)
            rows.append(row)
    return rows
