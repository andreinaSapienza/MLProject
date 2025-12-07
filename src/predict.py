import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ⚠️ MODIFICA QUI: Puntiamo al file "segreto" creato da train_model.py
DATA_FILE = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")

def predict_threats():
    print("--- SISTEMA DI RILEVAMENTO MINACCE (Simulazione Real-Time) ---")
    
    # 1. Carica il cervello (Modello + Scaler)
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Errore: Modello non trovato. Esegui prima train_model.py!")
        return

    print("Caricamento modello...", end=" ")
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Fatto.")

    # 2. Carica i nuovi dati (Dati mai visti dal training)
    if not os.path.exists(DATA_FILE):
        print(f"Errore: File holdout non trovato: {DATA_FILE}")
        print("Assicurati di aver eseguito il nuovo train_model.py che genera l'holdout set.")
        return

    print(f"Lettura dati dal 'Caveau' (Holdout Set): {DATA_FILE}")
    df_holdout = pd.read_csv(DATA_FILE)
    
    # --- SIMULAZIONE ---
    # Per la demo, prendiamo 20 pacchetti misti dal file holdout.
    # Cerchiamo di prenderne 10 Benigni e 10 Malevoli per vedere se li riconosce entrambi.
    
    try:
        # Preleva 10 benigni a caso (se ce ne sono abbastanza)
        benign_sample = df_holdout[df_holdout['Label'] == 0].sample(n=min(10, len(df_holdout[df_holdout['Label'] == 0])))
        # Preleva 10 malevoli a caso
        malicious_sample = df_holdout[df_holdout['Label'] == 1].sample(n=min(10, len(df_holdout[df_holdout['Label'] == 1])))
        
        # Unisci e mescola
        df_demo = pd.concat([benign_sample, malicious_sample]).sample(frac=1).reset_index(drop=True)
        
    except ValueError:
        # Se qualcosa va storto (es. file troppo piccolo), prendine 20 a caso
        df_demo = df_holdout.sample(n=min(20, len(df_holdout)))

    # Separiamo Features e Label Reale
    X_new = df_demo.drop(columns=['Label'])
    y_real = df_demo['Label']

    # 3. Pre-processing (USIAMO LO SCALER SALVATO)
    X_new_scaled = scaler.transform(X_new)

    # 4. Predizione
    print("\n--- Analisi Traffico in Corso ---")
    predictions = clf.predict(X_new_scaled)
    probs = clf.predict_proba(X_new_scaled)

    # 5. Stampa Risultati
    print(f"{'ID':<5} {'Predizione':<15} {'Sicurezza':<10} {'Realtà':<10} {'Esito'}")
    print("-" * 65)
    
    correct = 0
    for i, pred in enumerate(predictions):
        label_str = "🔴 MALWARE" if pred == 1 else "🟢 BENIGN"
        real_str = "Malware" if y_real[i] == 1 else "Benign"
        
        # Probabilità della classe predetta
        confidence = probs[i][pred] * 100
        
        # Check correttezza
        is_correct = (pred == y_real[i])
        status = "✅" if is_correct else "❌ ERRORE"
        if is_correct: correct += 1

        print(f"{i:<5} {label_str:<15} {confidence:>6.1f}%    {real_str:<10} {status}")

    print("-" * 65)
    print(f"Accuratezza su questo campione: {correct}/{len(df_demo)} ({correct/len(df_demo)*100:.1f}%)")
    
  # Statistica extra: accuratezza su TUTTO l'holdout set
    print("\n--- Verifica Scientifica (Intero Holdout Set) ---")
    X_full = df_holdout.drop(columns=['Label'])
    y_full = df_holdout['Label']
    
    # Trasforma i dati
    X_full_scaled = scaler.transform(X_full)
    
    # Fai la predizione su TUTTI i dati (non solo i 20 della demo)
    y_pred_full = clf.predict(X_full_scaled)
    
    # 1. Stampa la tabella (Classification Report)
    print("\nReport Dettagliato (Holdout Set):")
    print(classification_report(y_full, y_pred_full, target_names=['Benign', 'Malicious']))
    
    # 2. Stampa la Matrice di Confusione (Utile per vedere i numeri assoluti)
    print("Matrice di Confusione:")
    print(confusion_matrix(y_full, y_pred_full))

if __name__ == "__main__":
    predict_threats()