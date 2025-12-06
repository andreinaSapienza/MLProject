import pandas as pd
import os

# Percorsi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 1. Prendiamo un file MALICIOUS a caso
malicious_file = os.path.join(DATA_DIR, "malicious", "Raspberry_Binary.csv")

# 2. Prendiamo un file BENIGN a caso
benign_file = os.path.join(DATA_DIR, "benign", "Raspberry_download_benign.csv")

def check_mac(filepath, label):
    print(f"\n--- Analisi {label} ---")
    print(f"File: {os.path.basename(filepath)}")
    try:
        # Carichiamo solo le colonne dei MAC
        df = pd.read_csv(filepath, usecols=['Hw_src', 'HW_dst'])
        
        print(f"Totale righe: {len(df)}")
        print("MAC Sorgente più frequenti:")
        print(df['Hw_src'].value_counts().head(3))
        
        print("\nMAC Destinazione più frequenti:")
        print(df['HW_dst'].value_counts().head(3))
        
    except Exception as e:
        print(f"Errore lettura file: {e}")

if __name__ == "__main__":
    check_mac(malicious_file, "MALICIOUS")
    check_mac(benign_file, "BENIGN")