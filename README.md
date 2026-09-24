# 🧭 GraphSAGE – ein Gewicht für mich, ein anderes für meine Nachbarn

Viertes Stück der **Graph-Neural-Network-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning – und Nachfolger von
[gcn-demo](https://sebastianhanisch-gcn-demo.streamlit.app/) (GCN → GraphSAGE, GAT → GATv2, GIN → Graph Transformer; GIN und Graph Transformer sind noch nicht gebaut).

Vehikel **D "Liefergebiete"** wie im Vorgänger: Kunden in räumlichen Gebietstypen, vier verrauschte Merkmale, Graph der nächsten Nachbarn, wenige bekannte Etiketten, optional falsche Kanten. Ein **GCN** mittelt jeden Kunden zusammen mit seinen
Nachbarn und rechnet das Mittel mit **einer** Gewichtsmatrix um. **GraphSAGE** (Hamilton/Ying/Leskovec 2017) trennt beides: $H' = \mathrm{ReLU}(H W_s + (P H) W_n)$ – ein **eigenes Gewicht** $W_s$ für den Kunden, ein anderes $W_n$ für das Mittel
seiner Nachbarn – und darf im Training je Kunde nur eine **Stichprobe** der Nachbarn ziehen. Alle Daten sind erzeugt, das Netz ist von Grund auf in numpy geschrieben (Vorwärts- und Rückwärtsrechnung von Hand).

**Bezug zu OR:** Nachbarschaftsgraphen tragen Tourenplanung und Gebietszuschnitt; große Graphen lassen sich nur über Stichproben der Nachbarschaft trainieren.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie sah für GraphSAGE vor, dass es die Schwächen des GCN behebt: "transduktiv, braucht den ganzen Graphen". Die **Vorab-Messung hat das zum größten Teil widerlegt**:

1. **Induktiv ist auch das GCN.** Beide Netze haben Gewichte, die an keinem Knoten hängen; man trainiert auf der sichtbaren Hälfte des Gebiets und wendet auf die andere an (die Normierung des GCN berechnet man für den neuen Graphen neu).
   Beide verlieren durch den fehlenden Rest des Graphen ähnlich viel, und GraphSAGE bleibt hinter dem GCN (Experiment 3).
2. **GraphSAGE ist bei sauberen Kanten nicht besser, sondern meist schlechter als das GCN** – am stärksten bei verrauschten Merkmalen und wenigen Etiketten (Experiment 2). Der einzige Vorsprung: sehr geringes Rauschen.
3. Was sich **bestätigt** hat, ist die Rolle des **eigenen Gewichts**: bei falschen Kanten fällt das GCN unter das Netz ohne Nachbarn, GraphSAGE bleibt auf dessen Niveau – ein Boden, den die Ablation ("nur Nachbarn", eigenes Gewicht bei 0 eingefroren) als Ursache bestätigt
   (Experiment 1). Es ist eine **Versicherung**, kein Gewinn: über alle Stufen des Experiments ist GraphSAGE höchstens 1,1 Punkte besser als das bessere der beiden anderen Netze (nur bei sehr geringem Rauschen liegt es klar vor dem GCN, Experiment 2).
4. Die **Stichprobe** spart Rechenarbeit proportional, kostet aber Genauigkeit (Experiment 4).

## Modell

- **Vehikel** (`sg_scenario.py`): wie gcn-demo (200 Kunden, 3 Gebietstypen, 5 Etiketten je Typ, Rauschen 1,5, 5 nächste Nachbarn); dazu ein induktiver Split (nur eine Hälfte der Kunden sichtbar) und ein Schutz, dass bei kleinen Gebieten immer Kunden zur Prüfung übrig bleiben.
- **GraphSAGE** (`sg_algorithm.py`): Mittelwert-Aggregator, $P$ = Zeilenmittel über die Nachbarn ohne den Kunden; zwei Schichten (16 verdeckte Einheiten, ReLU, letzte Schicht ohne), Kreuzentropie auf den bekannten Kunden, Gewichtszerfall $5\cdot10^{-4}$ auf allen Matrizen,
  Adam (0,01), 200 Epochen, **kein Dropout, nichts am Test abgestimmt**. Stichprobe: je Schicht und Epoche höchstens $S$ zufällig gezogene Nachbarn je Kunde (ohne Zurücklegen); Auswertung mit allen Nachbarn.
- **Vergleiche:** GCN und MLP wie im Vorgänger (gleiche Größen); **GraphSAGE ohne eigenes Gewicht** ("nur Nachbarn": $W_s$ bei 0 eingefroren) als Ablation.

## Methodik

- **Handrechnungen:** Mittelwertmatrix auf einem Pfad und mit isoliertem Knoten; eine SAGE-Schicht auf dem Pfad 0–1–2 (Knoten 1: $(0,1) + 2\,(1{,}5;\,1) = (3,\,3)$); Nachrichtenzahlen der Stichprobe exakt aus den Graden.
- **Gegenproben:** Gleichwertigkeit mit der Verkettungsform $W\,[h_i \Vert \bar h_{\mathcal N(i)}]$; mit $W_s = W_n$ ein Mittel aus dem Kunden und dem Nachbarmittel; **mit $W_n = 0$ sind Verlust und Gradient exakt die des MLP** (unabhängige GCN-Rechnung);
  **Gradienten aller Gewichte gegen zentrale Differenzen** (1, 2, 3 Schichten, auch mit gezogenem $\tilde P$ – nicht symmetrisch), GCN-Gradienten; Stichprobe: höchstens $S$ echte Nachbarn je Zeile, Zeilensumme 1, bei $S \ge$ Grad gleich der Mittelwertmatrix,
  und **erwartungstreu** (Mittel über 3 000 Ziehungen weicht höchstens 0,06 von $P$ ab); beim induktiven Training haben die Merkmale nicht sichtbarer Kunden keinen Einfluss auf die Gewichte.
- **Statistik:** Experimente über 12 feste Seeds, Fehlerbalken = Standardfehler, Differenzen **gepaart je Seed**; die Zahl der Nachrichten wird gezählt (Kanten je Epoche über alle Schichten), **nicht die Rechenzeit gemessen**.
- **Literatur** (nicht nachgebaut): Hamilton, Ying, Leskovec 2017 ("Inductive representation learning on large graphs", NeurIPS); Kipf/Welling 2017 (GCN).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Standardfall (Seed 0) | Ein Einzelgebiet, saubere Kanten (Homophilie 90,8 %): GraphSAGE 91,9 %, GCN 94,1 %, MLP 48,1 % (Raten 50,3 %). | `test_standard_preset_both_graph_networks_beat_the_mlp_by_a_lot` |
| **Falsche Kanten** (12 Seeds; Homophilie 91 % → 38 %) | GraphSAGE minus GCN: 0 %: **−2,7 ± 1,0** (GCN in 11 von 12 Gebieten vorn); 20 %: −5,5 ± 2,8; 40 %: −8,1 ± 1,4; 60 %: −5,9 ± 2,8; 80 %: **+6,0 ± 3,1** (Homophilie 49 %); 100 %: **+10,5 ± 2,5** (GraphSAGE in 10 von 12 vorn; 56,6 % gegen 46,2 %). GraphSAGE minus MLP: +27,3 ± 2,8 bei sauberen Kanten, **−1,2 ± 2,2** bei zufälligen (MLP 57,8 %). | `test_wrong_experiment_sage_behind_gcn_when_clean_ahead_when_random`, `test_wrong_experiment_sage_holds_the_mlp_floor_and_needs_the_self_weight` |
| **Ist das eigene Gewicht die Ursache?** (Ablation) | Ohne eigenes Gewicht fällt GraphSAGE bei zufälligen Kanten auf **47,1 %** (mit: 56,6 %; MLP 57,8 %; GCN 46,2 %) und bei 80 % auf 44,5 % (mit: 59,0 %): der Boden beim MLP-Niveau kommt vom eigenen Gewicht. Der größte Vorsprung von GraphSAGE gegenüber dem besseren von GCN und MLP über alle Stufen beträgt **+1,1 Punkte**. | dieselben Tests |
| **Was kostet das eigene Gewicht?** (saubere Kanten, 12 Seeds) | Rauschen 0,5: **+3,0 ± 0,5** (GraphSAGE in 11 von 12 vorn); 1,0: −2,2 ± 1,8; 1,5: −2,7 ± 1,0; 2,5: **−9,8 ± 2,2**; 3,5: **−14,0 ± 2,5**. Etiketten je Typ 2: **−7,5 ± 2,0**; 3: −5,5 ± 1,7; 5: −2,7 ± 1,0; 10: −3,8 ± 1,3; 20: −1,7 ± 0,9. Ob es an den doppelt so vielen Gewichten oder daran liegt, dass das GCN den Kunden mit seinen Nachbarn mittelt und so das Rauschen dämpft, wurde nicht getrennt gemessen. | `test_cost_experiment_sage_wins_only_at_low_noise_and_falls_behind_with_noise`, `test_cost_experiment_sage_behind_gcn_at_every_label_level` |
| **Induktiv?** (400 Kunden, Hälfte sichtbar, 12 Seeds) | Saubere Kanten: GraphSAGE 84,2 %, GCN 89,4 % (−5,2 ± 1,5); Verlust gegenüber dem vollen Graphen im Training 2,2 ± 1,1 bzw. 3,0 ± 1,8 Punkte. 40 % falsche Kanten: 67,6 % gegen 69,8 % (−2,2 ± 1,0); Verlust 6,3 ± 1,8 bzw. 8,5 ± 2,6. Beide Netze verlieren ähnlich viel; das GCN ist **genauso induktiv**. | `test_inductive_experiment_gcn_is_just_as_inductive_and_ahead` |
| **Stichprobe** (dichter Graph, 20 nächste Nachbarn, Grad im Mittel 22,9; saubere Kanten) | Alle Nachbarn 87,7 %; 10: 86,5 % (**−1,3 ± 0,6**, 44 % der Nachrichten); 5: 83,5 % (−4,3 ± 1,0; 22 %); 2: 75,3 % (−12,5 ± 1,9; 9 %); 1: 69,3 % (−18,4 ± 2,2; 4 %). Mit 40 % falschen Kanten: 76,1 % alle; 71,7 % bei 10 (−4,5 ± 1,2); 62,4 % bei 1 (−13,7 ± 1,8). Die Arbeit sinkt im selben Verhältnis wie die Stichprobe, die Genauigkeit erst bei kleinen Stichproben deutlich. | `test_sample_experiment_message_shares_and_accuracy_cost` |
| Viele falsche Nachbarn (Preset, Seed 8, 80 %) | Homophilie 45,5 %: GraphSAGE 64,3 %, MLP 61,1 %, GCN 55,1 %. | `test_eighty_percent_wrong_edges_gcn_falls_below_the_mlp_and_sage_stays_near_or_above` |
| Nur zufällige Nachbarn (Preset, Seed 0, 100 %) | Homophilie 36,3 %: GraphSAGE 50,3 %, MLP 48,1 %, GCN 35,7 %. | `test_all_random_edges_sage_holds_the_mlp_level_gcn_below` |
| Stark verrauschte Merkmale (Preset, Seed 5) | Rauschen 3,0: GraphSAGE 69,7 %, GCN 82,7 %, MLP 45,9 %. | `test_noisy_preset_sage_far_behind_gcn` |
| Kaum Rauschen (Preset, Seed 3) | Rauschen 0,5: GraphSAGE 96,8 %, GCN 94,6 %, MLP 60,0 %. | `test_low_noise_preset_sage_close_to_or_ahead_of_gcn` |
| Dichter Graph, Stichprobe 2 (Preset, Seed 11) | 10 Nachbarn je Kunde (Grad im Mittel 11,6): GraphSAGE 81,6 % mit **800 Nachrichten** je Epoche gegen 85,4 % mit allen (**4.648**); GCN 90,8 %, MLP 60,0 %. | `test_dense_preset_sampling_message_counts_are_exact_and_accuracy_costs` |

Die Preset-Zeilen sind **Einzelgebiete** (Seeds so gewählt, dass die Klassen nicht stark ungleich groß sind und das Ergebnis nahe dem Median über zwölf Gebiete liegt); die Tests prüfen dort nur Strukturgrenzen und die exakt gezählten Nachrichten.
Belastbar sind die Mehr-Seed-Zeilen. Im Standardfall liegt GraphSAGE im Mittel über 12 Gebiete 2,7 Punkte hinter dem GCN.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Etiketten für doppelt so viele Gewichte** | Bei wenigen Etiketten und verrauschten Merkmalen liegt GraphSAGE hinter dem GCN (Experiment 2). | – |
| **Die Nachbarn sind entweder verlässlich oder egal** | Bei falschen Kanten hält das eigene Gewicht den Boden beim MLP-Niveau; wie viel ein einzelner Nachbar wert ist, kann GraphSAGE nicht unterscheiden – alle zählen im Mittel gleich. | GAT, GATv2 (Vorgänger) |
| **Mittelwert genügt als Aggregator** | Ein Mittel unterscheidet Nachbarschaften mit gleichem Durchschnitt nicht; hier nur der Mittelwert-Aggregator gemessen (kein Pooling, kein LSTM). | GIN |
| **Die Stichprobe ist erwartungstreu** | Sie schätzt das Nachbarmittel unverzerrt (Test), rauscht aber; bei kleinen Stichproben sinkt die Genauigkeit (Experiment 4). Gezählt werden Nachrichten, nicht Rechenzeit. | – |
| **Lokale Nachbarschaft genügt** | Information aus weit entfernten Kunden muss durch viele Schichten wandern. | Graph Transformer |
| **Erzeugte Daten, zwölf Gebiete je Messpunkt** | Ein Vehikel mit vier Merkmalen; die Zahlen gelten für diese Größen, Standardfehler sind groß. Nur zwei Schichten und die Standard-Hyperparameter gemessen; nichts abgestimmt. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, rund 3 Minuten wegen der Experimente): Kern per Handrechnung und Gegenprobe (Mittelwertmatrix, Stichprobe, SAGE-Schicht, Verkettungsform, geteilte Gewichte, Gradienten, Grenzfall MLP, Training, induktive Vorhersage), Vehikel, Auswertung
(Zeilenkonsistenz aller vier Experimente, Nachrichten exakt), Preset- und Permalink-Klemmen, AppTest-Rauchtests (jedes Preset, Kundenwahl mit Stichprobe, Extremwerte, vier Experimente auf Abruf) und `test_claims.py` (jede Zahl aus diesem README; Einzelgebiete nur mit
Strukturgrenzen, Mehr-Seed-Zahlen mit großzügigen Bändern).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `sg_constants.py` | Regler-Grenzen, Netz-Konstanten, Experiment-Seeds |
| `sg_presets.py` | Permalink/Presets-Mechanik |
| `sg_scenario.py` | Vehikel D, induktiver Split, induzierter Teilgraph |
| `sg_algorithm.py` | Mittelwertmatrix, Stichprobe, GraphSAGE/GCN/MLP (Schichten, Gradienten, Adam) |
| `sg_evaluation.py` | Analyse, vier Experimente |
| `sg_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Pooling- und LSTM-Aggregator, Dropout, Feinabstimmung der Hyperparameter, Rechenzeitmessung großer Graphen.
- Die Ursachenforschung, warum GraphSAGE bei verrauschten Merkmalen hinter dem GCN liegt.
- Die übrigen Stücke der Linie (GIN, Graph Transformer); ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
