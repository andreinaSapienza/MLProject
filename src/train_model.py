import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import joblib
import os

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "final_dataset_windowed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def train_and_evaluate():
    print(f"--- 1. Caricamento Dataset ---")
    if not os.path.exists(DATA_FILE):
        print("ERRORE: File non trovato!")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Totale finestre: {len(df)}")
    
    # Controllo distribuzione classi totale
    print(f"Distribuzione Totale: {df['Label'].value_counts().to_dict()}")

    X = df.drop(columns=['Label'])
    y = df['Label']

    # --- 2. BLOCK-BASED SPLIT (La Soluzione) ---
    # Creiamo un "Group ID". Ogni 100 righe consecutive sono un gruppo.
    # Questo mantiene intatta la rolling window locale, ma ci permette di mescolare i gruppi.
    BLOCK_SIZE = 100
    # Esempio: righe 0-99 -> Gruppo 0, righe 100-199 -> Gruppo 1, ecc.
    groups = np.arange(len(df)) // BLOCK_SIZE
    
    print(f"\nGenerazione blocchi (dimensione {BLOCK_SIZE})...")
    
    # GroupShuffleSplit mescola i gruppi, non le singole righe!
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"Training Set: {len(X_train)} righe")
    print(f"Test Set:     {len(X_test)} righe")
    
    # VERIFICA CRUCIALE
    unique_classes = y_test.unique()
    print(f"Classi nel Test Set: {unique_classes}")
    if len(unique_classes) < 2:
        print("⚠️ ERRORE: Ancora una sola classe nel test set. Il dataset potrebbe essere troppo sbilanciato o ordinato in macro-blocchi enormi.")
        return

    # --- 3. NORMALIZZAZIONE ---
    print("\n--- Normalizzazione Features ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. ADDESTRAMENTO ---
    print("\n--- Avvio Addestramento Random Forest ---")
    clf = RandomForestClassifier(n_estimators=100, 
                                 random_state=42, 
                                 n_jobs=-1,
                                 class_weight='balanced') # Importante per bilanciare le classi
    clf.fit(X_train_scaled, y_train)
    print("Addestramento completato!")

    # --- 5. VALUTAZIONE ---
    print("\n--- Valutazione sul Test Set (Block-Split) ---")
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuratezza: {acc:.4f}")

    print("\nMatrice di Confusione:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReport Dettagliato:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))

    # --- 6. SALVATAGGIO ---
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✅ Modello salvato correttamente in {MODEL_DIR}")

    # --- 7. FEATURE IMPORTANCE ---
    print("\n--- Top 5 Features ---")
    importances = clf.feature_importances_
    features_desc = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    for name, imp in features_desc[:5]:
        print(f"{name:15}: {imp:.4f}")

if __name__ == "__main__":
    train_and_evaluate()