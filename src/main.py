import os
import sys
import pandas as pd

# Setup path per importare utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.loader import load_and_process_data, filter_by_mac

# Percorsi dinamici
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_traffic.csv")

def get_top_mac(df):
    """
    Restituisce il MAC più frequente nel dataframe, 
    contando sia le apparizioni come Sorgente che come Destinazione.
    """
    if df.empty:
        return None
        
    # Uniamo le due colonne in una sola serie lunghissima
    all_macs = pd.concat([df['Src_MAC'], df['Dst_MAC']])
    
    # Troviamo il valore più frequente (il dispositivo "protagonista")
    mode_result = all_macs.mode()
    
    return mode_result[0] if len(mode_result) > 0 else None

def main():
    # 1. Carica tutti i dati
    try:
        df = load_and_process_data(DATA_DIR)
    except Exception as e:
        print(f"Errore Fatale: {e}")
        return

    # 2. Rileva i MAC Address (SEPARATAMENTE per Benign e Malicious)
    print("\n--- Rilevamento MAC Address Multipli (Src + Dst) ---")
    
    # Prendi il MAC principale dai dati Benign (Label 0)
    df_benign = df[df['label'] == 0]
    mac_benign = get_top_mac(df_benign)
    print(f"MAC Rilevato (Scenario Benign):    {mac_benign}")
    
    # Prendi il MAC principale dai dati Malicious (Label 1)
    df_malicious = df[df['label'] == 1]
    mac_malicious = get_top_mac(df_malicious)
    print(f"MAC Rilevato (Scenario Malicious): {mac_malicious}")
    
    # Creiamo la lista dei MAC validi (escludendo eventuali None)
    valid_macs = [m for m in [mac_benign, mac_malicious] if m is not None]
    
    # Rimuovi duplicati se per caso sono uguali
    valid_macs = list(set(valid_macs))

    if not valid_macs:
        print("Errore: Impossibile rilevare MAC address.")
        return

    # 3. Filtra usando la lista di MAC
    df_raspberry = filter_by_mac(df, valid_macs)
    
    # 4. Salva
    print(f"\n>> Salvataggio in: {OUTPUT_FILE}")
    df_raspberry.to_csv(OUTPUT_FILE, index=False)
    print("Fase 1 completata con successo!")

if __name__ == "__main__":
    main()