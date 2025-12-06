import pandas as pd
import os

# --- COSTRUZIONE DEL PERCORSO SICURO ---
# 1. Trova dov'è questo file (check_labels.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Torna indietro di una cartella (da 'src' a 'LabML')
base_dir = os.path.dirname(current_dir)
# 3. Costruisci il percorso verso i dati
data_file = os.path.join(base_dir, "data", "processed_traffic.csv")

print(f"Lettura file da: {data_file}")

try:
    df = pd.read_csv(data_file)
    
    print("\n--- 1. CONTEGGIO TOTALE CLASSI ---")
    # Questo risponde alla tua domanda: "Ci sono solo zeri?"
    counts = df['label'].value_counts()
    print(counts)
    
    print("\nLegenda:")
    print("0 = Benign (Traffico normale)")
    print("1 = Malicious (Attacchi)")

    print(f"\nPercentuale Malicious: {counts.get(1, 0) / len(df) * 100:.2f}%")

    print("\n--- 2. ANTEPRIMA DATI MALICIOUS ---")
    # Mostriamo solo le righe dove label == 1 per dimostrare che esistono
    malicious_sample = df[df['label'] == 1].head(5)
    
    if not malicious_sample.empty:
        print("Ecco 5 righe di traffico malevolo trovate nel file:")
        print(malicious_sample[['Timestamp', 'Src_MAC', 'label']])
    else:
        print("ATTENZIONE: Non ho trovato nessuna riga con label=1!")

except FileNotFoundError:
    print(f"ERRORE: Il file non esiste nel percorso: {data_file}")
    print("Assicurati di aver eseguito prima main.py!")