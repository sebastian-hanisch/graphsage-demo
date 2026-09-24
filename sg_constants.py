"""Konstanten der GraphSAGE-Demo: Vehikel D "Liefergebiete" wie im GCN-Vorgänger (Kunden im Gebiet, Graph der räumlichen Nachbarn, vier Merkmale, Etikett = Gebietstyp), Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

AREA = 100.0
CLASS_NAMES = ("Innenstadt", "Vorstadt", "Ländlich", "Gewerbegebiet")
FEATURE_NAMES = ("Stopps je Stunde", "Parksuche (min)", "Ladegewicht (kg)", "Zeitfenster-Enge")
N_FEATURES = len(FEATURE_NAMES)

N_MIN, N_MAX, N_STEP, DEFAULT_N = 60, 400, 20, 200
CLASSES_MIN, CLASSES_MAX, DEFAULT_CLASSES = 2, 4, 3
LABELS_MIN, LABELS_MAX, DEFAULT_LABELS = 2, 20, 5
NEIGHBORS_MIN, NEIGHBORS_MAX, DEFAULT_NEIGHBORS = 2, 10, 5
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 0.5, 4.0, 0.25, 1.5
WRONG_MIN, WRONG_MAX, WRONG_STEP, DEFAULT_WRONG = 0.0, 1.0, 0.05, 0.0
LAYERS_MIN, LAYERS_MAX, DEFAULT_LAYERS = 1, 4, 2
SAMPLE_OPTIONS = (0, 1, 2, 3, 5, 8)                 # Nachbarn je Kunde und Schicht im Training; 0 = alle
DEFAULT_SAMPLE = 0
HIDDEN = 16
EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# --- Experimente (feste Seeds) --------------------------------------------------------------------------------------------------------------

EXP_SEEDS = tuple(range(12))
WRONG_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
NOISE_LEVELS = (0.5, 1.0, 1.5, 2.5, 3.5)
LABEL_LEVELS = (2, 3, 5, 10, 20)
IND_N = 400                                          # Kunden im Gesamtgebiet; die Hälfte ist im Training sichtbar
IND_SEEN = 0.5
IND_WRONG_LEVELS = (0.0, 0.4)
DENSE_K = 20                                         # Nachbarn je Kunde im Stichproben-Experiment (dichter Graph)
SAMPLE_LEVELS = (1, 2, 5, 10, 0)
SAMPLE_WRONG_LEVELS = (0.0, 0.4)
