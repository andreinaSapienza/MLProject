import pandas as pd
import joblib
import os
import numpy as np

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# File da analizzare (Usiamo lo stesso windowed per test, ma idealmente sarebbe nuovo)
DATA_FILE = os.path.join(BASE_DIR, "data", "final_dataset_windowed.csv")

def predict_threats():
    print("--- SISTEMA DI RILEVAMENTO MINACCE ---")
    
    # 1. Carica il cervello (Modello + Scaler)
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Errore: Modello non trovato. Addestralo prima!")
        return

    print("Caricamento modello...", end=" ")
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Fatto.")

    # 2. Carica i nuovi dati (Simulazione)
    # Preleviamo 50 campioni casuali dal dataset per simulare traffico in arrivo
    print(f"Lettura dati da: {DATA_FILE}")
    df_all = pd.read_csv(DATA_FILE)
    
    # Prendiamo 10 esempi Benigni e 10 Malevoli a caso per vedere se li riconosce
    df_benign = df_all[df_all['Label'] == 0].sample(10)
    df_malicious = df_all[df_all['Label'] == 1].sample(10)
    
    # Uniamo e mescoliamo
    df_test = pd.concat([df_benign, df_malicious]).sample(frac=1).reset_index(drop=True)
    
    # Separiamo Features e Label Reale (per confronto)
    X_new = df_test.drop(columns=['Label'])
    y_real = df_test['Label']

    # 3. Pre-processing (Fondamentale: usare lo STESSO scaler del training)
    X_new_scaled = scaler.transform(X_new)

    # 4. Predizione
    print("\n--- Analisi Traffico in Corso ---")
    predictions = clf.predict(X_new_scaled)
    probs = clf.predict_proba(X_new_scaled) # Probabilità (sicurezza del modello)

    # 5. Stampa Risultati
    print(f"{'ID':<5} {'Predizione':<15} {'Sicurezza':<10} {'Realtà':<10} {'Esito'}")
    print("-" * 60)
    
    correct = 0
    for i, pred in enumerate(predictions):
        label_str = "🔴 MALWARE" if pred == 1 else "🟢 BENIGN"
        real_str = "Malware" if y_real[i] == 1 else "Benign"
        
        # Probabilità della classe predetta
        confidence = probs[i][pred] * 100
        
        # Check correttezza
        is_correct = (pred == y_real[i])
        status = "✅" if is_correct else "❌ ALLARME ERRORE"
        if is_correct: correct += 1

        print(f"{i:<5} {label_str:<15} {confidence:>6.1f}%    {real_str:<10} {status}")

    print("-" * 60)
    print(f"Accuratezza su questo campione casuale: {correct}/{len(df_test)} ({correct/len(df_test)*100:.1f}%)")

if __name__ == "__main__":
    predict_threats()