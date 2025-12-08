import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import joblib
import os

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "final_dataset_windowed.csv")

# File di output
HOLDOUT_FILE = os.path.join(BASE_DIR, "data", "holdout_dataset.csv") 
DEV_FILE = os.path.join(BASE_DIR, "data", "dev_set.csv")  # <--- NUOVO FILE

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def train_and_evaluate():
    print(f"--- 1. Caricamento Dataset ---")
    if not os.path.exists(DATA_FILE):
        print("ERRORE: File non trovato!")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Totale finestre originali: {len(df)}")

    # --- 2. CREAZIONE HOLDOUT SET & DEV SET ---
    # Mettiamo da parte il 10% dei dati per predict.py. 
    BLOCK_SIZE = 100
    groups = np.arange(len(df)) // BLOCK_SIZE
    
    # Primo split: 90% Dev (Lavoro), 10% Holdout (Segreto)
    splitter_holdout = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=999)
    train_dev_idx, holdout_idx = next(splitter_holdout.split(df, groups=groups))
    
    df_holdout = df.iloc[holdout_idx]
    df_working = df.iloc[train_dev_idx] # Questo è il 90% (Dev Set)
    
    # Salviamo il file segreto (Holdout)
    print(f"\nSalvataggio Holdout Set (Mai visto dal modello): {len(df_holdout)} righe")
    df_holdout.to_csv(HOLDOUT_FILE, index=False)

    # --- ✅ MODIFICA: SALVATAGGIO DEV SET ---
    # Salviamo il dataset di lavoro su disco per poterlo usare con la Rete Neurale
    print(f"Salvataggio Dev Set (Per training DL e RF): {len(df_working)} righe")
    df_working.to_csv(DEV_FILE, index=False)
    # ----------------------------------------

    # --- 3. PREPARAZIONE TRAINING RANDOM FOREST ---
    # Usiamo df_working che ora è caricato in memoria (è lo stesso di dev_set.csv)
    X = df_working.drop(columns=['Label'])
    y = df_working['Label']
    
    # Ricalcoliamo i gruppi sul nuovo dataset ridotto
    groups_working = np.arange(len(df_working)) // BLOCK_SIZE

    # --- 4. BLOCK SPLIT INTERNO (Train vs Validation) ---
    # Dividiamo il dev set: 80% Train, 20% Validation
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups_working))
    
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"\nTraining Set Reale (RF): {len(X_train)} righe")
    print(f"Validation Set Reale (RF): {len(X_test)} righe")

    # --- 5. NORMALIZZAZIONE ---
    print("\n--- Normalizzazione ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 6. ADDESTRAMENTO ---
    print("\n--- Avvio Addestramento Random Forest ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    # Valutazione interna
    print("\n--- Risultati sul Validation Set (RF) ---")
    y_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))

    # --- 7. SALVATAGGIO MODELLO ---
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✅ Modello RF salvato.")

if __name__ == "__main__":
    train_and_evaluate()