import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# MODIFICA QUI: Il percorso ora è .../LabML/results/plots
IMG_DIR = os.path.join(BASE_DIR, "results", "plots")

def plot_confusion_matrix():
    print("--- GENERAZIONE GRAFICO MATRICE DI CONFUSIONE ---")
    
    # 1. Caricamento Dati e Modello
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_PATH):
        print(f"Errore: Mancano i file dati o il modello.\nControlla: {DATA_FILE}")
        return

    print("Caricamento...", end=" ")
    df = pd.read_csv(DATA_FILE)
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Fatto.")

    # 2. Preparazione
    X = df.drop(columns=['Label'])
    y_true = df['Label']
    X_scaled = scaler.transform(X)

    # 3. Predizione
    y_pred = clf.predict(X_scaled)

    # 4. Calcolo Matrice
    cm = confusion_matrix(y_true, y_pred)
    
    # Definiamo le etichette
    labels = ['Benign', 'Malicious']
    
    # --- CREAZIONE GRAFICO ---
    plt.figure(figsize=(8, 6))
    
    # Disegna la heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 14})

    plt.ylabel('Realtà (True Label)', fontsize=12)
    plt.xlabel('Predizione (Predicted Label)', fontsize=12)
    plt.title('Matrice di Confusione - Random Forest (Holdout Set)', fontsize=14)

    # Crea le cartelle se non esistono (crea sia 'results' che 'plots')
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    # Salva
    save_path = os.path.join(IMG_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Grafico salvato in: {save_path}")
    print("Ora hai una cartella 'results/plots' pronta per la tesi!")
    
    # Mostra a video solo se possibile
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    plot_confusion_matrix()