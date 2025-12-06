import pandas as pd
import glob
import os

def load_and_process_data(data_path):
    print(f"--- Inizio caricamento dati da: {data_path} ---")
    
    # Colonne che ci servono dal file originale
    COLS_TO_USE = ['Time', 'Length', 'Hw_src', 'HW_dst']
    
    # Mappa per rinominare le colonne
    RENAME_MAP = {
        'Time': 'Timestamp',
        'Length': 'Packet_Length',
        'Hw_src': 'Src_MAC',
        'HW_dst': 'Dst_MAC'
    }

    # --- 1. CARICAMENTO MALICIOUS (Label = 1) ---
    # Nota: Aggiungiamo la cartella "malicious" nel path
    malicious_patterns = [
        "Raspberry_Binary.csv", 
        "Raspberry_Webmine*.csv"
    ]
    
    df_list = []
    
    print(">> Caricamento MALICIOUS...")
    for pattern in malicious_patterns:
        # QUI LA MODIFICA: data_path + "malicious" + pattern
        search_path = os.path.join(data_path, "malicious", pattern)
        files = glob.glob(search_path)
        
        # Debug: se non trova file, lo stampiamo
        if not files:
            print(f"   [WARN] Nessun file trovato per: {search_path}")

        for f in files:
            try:
                df = pd.read_csv(f, usecols=COLS_TO_USE, header=0)
                df = df.rename(columns=RENAME_MAP)
                df['label'] = 1
                df_list.append(df)
                print(f"   [OK] {os.path.basename(f)}: {len(df)} righe")
            except ValueError as ve:
                print(f"   [ERR] Colonne mancanti in {os.path.basename(f)}: {ve}")
            except Exception as e:
                print(f"   [ERR] {os.path.basename(f)}: {e}")

    # --- 2. CARICAMENTO BENIGN (Label = 0) ---
    # Nota: Aggiungiamo la cartella "benign" nel path
    print(">> Caricamento BENIGN...")
    benign_pattern = "Raspberry_*benign.csv"
    
    # QUI LA MODIFICA: data_path + "benign" + pattern
    search_path = os.path.join(data_path, "benign", benign_pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"   [WARN] Nessun file trovato per: {search_path}")
    
    for f in files:
        try:
            df = pd.read_csv(f, usecols=COLS_TO_USE, header=0)
            df = df.rename(columns=RENAME_MAP)
            df['label'] = 0
            df_list.append(df)
            print(f"   [OK] {os.path.basename(f)}: {len(df)} righe")
        except Exception as e:
            print(f"   [ERR] {os.path.basename(f)}: {e}")

    # --- 3. UNIONE ---
    if not df_list:
        raise ValueError("Errore critico: Nessun dato caricato. Controlla i percorsi e i nomi dei file.")

    df_total = pd.concat(df_list, ignore_index=True)
    
    # Pulizia Timestamp
    df_total['Timestamp'] = pd.to_numeric(df_total['Timestamp'], errors='coerce')
    df_total = df_total.dropna(subset=['Timestamp'])
    df_total = df_total.sort_values('Timestamp')
    
    return df_total

def filter_by_mac(df, target_macs):
    """
    Filtra il dataframe mantenendo i pacchetti dove Src O Dst
    sono presenti nella lista target_macs.
    """
    # Se target_macs è una stringa singola, trasformala in lista
    if isinstance(target_macs, str):
        target_macs = [target_macs]
        
    print(f"\n>> Filtraggio per i Device: {target_macs}")
    initial = len(df)
    
    # Controlla se Src O Dst sono nella lista dei MAC validi
    mask = (df['Src_MAC'].isin(target_macs)) | (df['Dst_MAC'].isin(target_macs))
    df_filtered = df[mask].copy()
    
    print(f"   Righe originali: {initial}")
    print(f"   Righe filtrate:  {len(df_filtered)}")
    return df_filtered