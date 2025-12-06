import pandas as pd
import numpy as np
import os

# Percorsi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_traffic.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "final_dataset_windowed.csv")

def calculate_features(df, window_size=10):
    print(f"--- Inizio Windowing (Finestra: {window_size} pacchetti) ---")
    
    # 1. Calcola il Delta Time (tempo tra un pacchetto e l'altro)
    df['Delta_Time'] = df['Timestamp'].diff().fillna(0)
    
    # 2. Creiamo le finestre scorrevoli (Rolling Window)
    # L'oggetto 'rolling' guarda le ultime 10 righe
    print("Calcolo statistiche (Media, Varianza, Min, Max)...")
    
    # Statistiche TEMPORALI (Comportamento veloce vs lento)
    time_mean = df['Delta_Time'].rolling(window=window_size).mean()
    time_var  = df['Delta_Time'].rolling(window=window_size).var().fillna(0)
    
    # Statistiche DIMENSIONALI (Pacchetti grandi vs piccoli)
    len_mean = df['Packet_Length'].rolling(window=window_size).mean()
    len_min  = df['Packet_Length'].rolling(window=window_size).min()
    len_max  = df['Packet_Length'].rolling(window=window_size).max()
    len_var  = df['Packet_Length'].rolling(window=window_size).var().fillna(0)
    
    # Gestione LABEL: Se nella finestra c'è un attacco, la finestra è sospetta.
    # Usiamo max(): se c'è anche solo un '1' nelle ultime 10 righe, l'etichetta diventa 1.
    labels = df['label'].rolling(window=window_size).max()
    
    # 3. Creiamo il DataFrame finale
    df_features = pd.DataFrame({
        'Time_Mean': time_mean,
        'Time_Var': time_var,
        'Length_Mean': len_mean,
        'Length_Min': len_min,
        'Length_Max': len_max,
        'Length_Var': len_var,
        'Label': labels
    })
    
    # Rimuoviamo le prime 9 righe che contengono NaN (perché la finestra non era piena)
    df_features = df_features.dropna()
    
    # Assicuriamoci che la label sia un numero intero (0 o 1)
    df_features['Label'] = df_features['Label'].astype(int)
    
    return df_features

def main():
    if not os.path.exists(DATA_FILE):
        print("Errore: processed_traffic.csv non trovato. Esegui prima main.py")
        return

    print(f"Caricamento dati da: {DATA_FILE}")
    df_raw = pd.read_csv(DATA_FILE)
    
    # Eseguiamo il feature engineering
    df_final = calculate_features(df_raw, window_size=10)
    
    print("\n--- Risultato ---")
    print(f"Righe Originali (Pacchetti): {len(df_raw)}")
    print(f"Righe Finali (Finestre):     {len(df_final)}")
    
    # Controllo rapido sulle classi
    counts = df_final['Label'].value_counts()
    print(f"\nDistribuzione Classi nel Dataset Finale:\n{counts}")
    
    print(f"\nSalvataggio in: {OUTPUT_FILE}")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("Fatto! Dataset pronto per il Machine Learning.")

if __name__ == "__main__":
    main()