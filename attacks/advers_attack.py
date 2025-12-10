import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import warnings

# Ignora warning inutili di sklearn
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Seed per riproducibilità (PRIMA DI TUTTO!)
np.random.seed(42)
torch.manual_seed(42)

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT_FILE = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")
RF_MODEL = os.path.join(BASE_DIR, "models", "rf_model.pkl")
RF_SCALER = os.path.join(BASE_DIR, "models", "scaler.pkl")
DL_MODEL = os.path.join(BASE_DIR, "models", "mlp_model.pth")
DL_SCALER = os.path.join(BASE_DIR, "models", "scaler_dl.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results/plots")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- DEFINIZIONE RETE NEURALE ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# --- ADVERSARIAL ATTACKS ---

def timing_jitter_attack(df, epsilon=0.05):
    """
    Attack 1: Aggiunge rumore gaussiano ai timing.
    Simula: sleep() randomizzato nel malware.
    """
    df_adv = df.copy()
    noise_time_mean = np.random.normal(0, epsilon * df['Time_Mean'].mean(), len(df))
    noise_time_var = np.abs(np.random.normal(0, epsilon * df['Time_Var'].mean(), len(df)))
    
    df_adv['Time_Mean'] = np.clip(df['Time_Mean'] + noise_time_mean, 0.001, None)
    df_adv['Time_Var'] = df['Time_Var'] + noise_time_var
    return df_adv

def packet_padding_attack(df, max_padding=50):
    """
    Attack 2: Aggiunge padding ai pacchetti.
    Simula: Offuscamento con byte spazzatura.
    """
    df_adv = df.copy()
    padding = np.random.uniform(0, max_padding, len(df))
    
    df_adv['Length_Mean'] = np.clip(df['Length_Mean'] + padding, 0, 1500)
    df_adv['Length_Min'] = np.clip(df['Length_Min'] + padding * 0.5, 0, 1500)
    df_adv['Length_Max'] = np.clip(df['Length_Max'] + padding, 0, 1500)
    df_adv['Length_Var'] = df['Length_Var'] + padding * 0.3
    return df_adv

def dummy_traffic_injection(df, injection_rate=0.3):
    """
    Attack 3: Mescola statistiche malicious con pattern benign.
    Simula: Mimicry attack (mimetismo).
    """
    df_adv = df.copy()
    n_inject = int(len(df) * injection_rate)
    
    benign_stats = {
        'Time_Mean': 0.12, 'Time_Var': 0.05,
        'Length_Mean': 200, 'Length_Var': 400
    }
    
    if n_inject > 0:
        inject_indices = np.random.choice(len(df), n_inject, replace=False)
        alpha = 0.6
        
        for col in ['Time_Mean', 'Time_Var', 'Length_Mean', 'Length_Var']:
            if col in df.columns:
                df_adv.loc[inject_indices, col] = (
                    alpha * df.loc[inject_indices, col] + 
                    (1-alpha) * benign_stats[col]
                )
    return df_adv

def burst_shaping_attack(df, burst_prob=0.2, burst_factor=3.0):
    """
    Attack 4: Crea burst irregolari di pacchetti.
    Simula: Traffico a raffica per nascondere periodicità.
    """
    df_adv = df.copy()
    mask = np.random.rand(len(df)) < burst_prob
    
    # Usa .loc[] per evitare SettingWithCopyWarning
    df_adv.loc[mask, 'Time_Var'] = df_adv.loc[mask, 'Time_Var'] * burst_factor
    df_adv.loc[mask, 'Length_Var'] = df_adv.loc[mask, 'Length_Var'] * burst_factor
    return df_adv

# --- EVALUATION ---

def evaluate_model_robustness(model_type='rf'):
    print(f"\n{'='*70}")
    print(f"   TEST ADVERSARIAL ROBUSTNESS - {model_type.upper()}")
    print(f"{'='*70}\n")
    
    # Verifica file esistenza
    if not os.path.exists(HOLDOUT_FILE):
        print(f"[ERROR] {HOLDOUT_FILE} non trovato!")
        print("   Esegui prima: python3 src/train_model.py")
        return [], 0
    
    # Caricamento dati
    df_holdout = pd.read_csv(HOLDOUT_FILE)
    df_malicious = df_holdout[df_holdout['Label'] == 1].copy()
    df_benign = df_holdout[df_holdout['Label'] == 0].copy()
    
    print(f"Holdout totale: {len(df_holdout)} campioni")
    print(f"  - Benign:     {len(df_benign)}")
    print(f"  - Malicious:  {len(df_malicious)}")
    
    # Caricamento modello
    if model_type == 'rf':
        # Verifica esistenza modello RF
        if not os.path.exists(RF_MODEL) or not os.path.exists(RF_SCALER):
            print(f"[ERROR] Modello RF non trovato!")
            print("   Esegui prima: python3 src/train_model.py")
            return [], 0
        
        model = joblib.load(RF_MODEL)
        scaler = joblib.load(RF_SCALER)
        
        def predict(df):
            X = scaler.transform(df.drop(columns=['Label']).values)
            return model.predict(X)
    
    else:  # DL
        # Verifica esistenza modello DL
        if not os.path.exists(DL_MODEL) or not os.path.exists(DL_SCALER):
            print(f"[ERROR] Modello DL non trovato!")
            print("   Esegui prima: python3 src/train_dl.py")
            return [], 0
        
        scaler = joblib.load(DL_SCALER)
        dummy_X = df_holdout.drop(columns=['Label']).values
        model = SimpleMLP(input_dim=dummy_X.shape[1])
        model.load_state_dict(torch.load(DL_MODEL, weights_only=True))
        model.eval()
        
        def predict(df):
            X = scaler.transform(df.drop(columns=['Label']).values)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                preds = (model(X_tensor) > 0.5).cpu().numpy().flatten()
            return preds.astype(int)
    
    # --- BASELINE (No Attack) ---
    print("\n--- BASELINE (No Attack) ---")
    y_pred_base = predict(df_holdout)
    y_true = df_holdout['Label']
    
    tp = ((y_pred_base == 1) & (y_true == 1)).sum()
    fn = ((y_pred_base == 0) & (y_true == 1)).sum()
    baseline_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Baseline Recall: {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
    print(f"Malicious rilevati: {tp}/{tp+fn}")
    
    # --- ADVERSARIAL ATTACKS (LISTA PULITA) ---
    attacks = [
        ("Timing Jitter (5%)", lambda df: timing_jitter_attack(df, 0.05)),
        ("Timing Jitter (10%)", lambda df: timing_jitter_attack(df, 0.10)),
        ("Timing Jitter (20%)", lambda df: timing_jitter_attack(df, 0.20)),
        ("Packet Padding (30 bytes)", lambda df: packet_padding_attack(df, 30)),
        ("Packet Padding (50 bytes)", lambda df: packet_padding_attack(df, 50)),
        ("Packet Padding (100 bytes)", lambda df: packet_padding_attack(df, 100)),
        ("Dummy Traffic (20%)", lambda df: dummy_traffic_injection(df, 0.20)),
        ("Dummy Traffic (30%)", lambda df: dummy_traffic_injection(df, 0.30)),
        ("Burst Shaping (20%)", lambda df: burst_shaping_attack(df, 0.20, 3.0)),
        ("Burst Shaping (40%)", lambda df: burst_shaping_attack(df, 0.40, 4.0)),
    ]
    
    results = []
    
    print(f"\n{'Attack':<30} {'Recall':<12} {'Drop':<12} {'Status'}")
    print("-" * 70)
    
    for name, func in attacks:
        # Applica attacco SOLO a malicious
        df_mal_adv = func(df_malicious)
        
        # Ricostruisci dataset completo
        df_test = pd.concat([df_benign, df_mal_adv], ignore_index=True)
        
        # Predici
        y_pred = predict(df_test)
        y_true_adv = df_test['Label']
        
        # Calcola recall
        tp = ((y_pred == 1) & (y_true_adv == 1)).sum()
        fn = ((y_pred == 0) & (y_true_adv == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        drop = baseline_recall - recall
        drop_pct = (drop / baseline_recall * 100) if baseline_recall > 0 else 0
        
        # Status text
        if drop < 0.02:
            status = "[OK] Resistente"
        elif drop < 0.05:
            status = "[!] Leggero calo"
        else:
            status = "[ALERT] Efficace"
        
        print(f"{name:<30} {recall:.4f}       {drop:+.4f}       {status}")
        
        results.append({
            'attack': name,
            'recall': recall,
            'drop': drop,
            'drop_pct': drop_pct
        })
    
    # Statistiche finali
    avg_recall = np.mean([r['recall'] for r in results])
    avg_drop = np.mean([r['drop'] for r in results])
    
    print("\n" + "="*70)
    print(f"ROBUSTEZZA MEDIA: Recall = {avg_recall:.4f} | Drop = {avg_drop:.4f} ({avg_drop/baseline_recall*100:.1f}%)")
    print("="*70)
    
    return results, baseline_recall

# --- MAIN ---
def main():
    print("="*70)
    print("   ADVERSARIAL ROBUSTNESS TEST")
    print("   Confronto: Random Forest vs Deep Learning")
    print("="*70)
    
    # Test RF
    rf_res, rf_base = evaluate_model_robustness('rf')
    
    if not rf_res:
        print("\n[ERROR] Test RF fallito. Interrompo.")
        return
    
    # Test DL
    dl_res, dl_base = evaluate_model_robustness('dl')
    
    if not dl_res:
        print("\n[ERROR] Test DL fallito. Interrompo.")
        return
    
    # --- CONFRONTO FINALE ---
    print("\n" + "="*70)
    print("   CONFRONTO ROBUSTEZZA: RF vs DL")
    print("="*70)
    
    rf_avg_drop = np.mean([r['drop'] for r in rf_res])
    dl_avg_drop = np.mean([r['drop'] for r in dl_res])
    
    print(f"\n{'Modello':<15} {'Baseline Recall':<20} {'Avg Drop':<15} {'Robustezza'}")
    print("-"*70)
    print(f"{'Random Forest':<15} {rf_base:.4f}               {rf_avg_drop:.4f}          {'[+] Piu robusto' if rf_avg_drop < dl_avg_drop else '[-] Meno robusto'}")
    print(f"{'Deep Learning':<15} {dl_base:.4f}               {dl_avg_drop:.4f}          {'[+] Piu robusto' if dl_avg_drop < rf_avg_drop else '[-] Meno robusto'}")
    
    # --- VISUALIZZAZIONE ---
    print("\n--- Generazione grafico comparativo ---")
    
    attack_names = [r['attack'] for r in rf_res]
    rf_recalls = [r['recall'] for r in rf_res]
    dl_recalls = [r['recall'] for r in dl_res]
    
    plt.figure(figsize=(16, 8))
    
    x = np.arange(len(attack_names))
    width = 0.35
    
    plt.bar(x - width/2, rf_recalls, width, label='Random Forest', 
            color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.bar(x + width/2, dl_recalls, width, label='Deep Learning', 
            color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Baseline lines
    plt.axhline(y=rf_base, color='#3498db', linestyle='--', alpha=0.5, 
                linewidth=2, label=f'RF Baseline ({rf_base:.2%})')
    plt.axhline(y=dl_base, color='#e74c3c', linestyle='--', alpha=0.5, 
                linewidth=2, label=f'DL Baseline ({dl_base:.2%})')
    
    # Annotazioni percentuali
    for i, (rf_val, dl_val) in enumerate(zip(rf_recalls, dl_recalls)):
        plt.text(i - width/2, rf_val + 0.005, f'{rf_val:.1%}', 
                ha='center', fontsize=7, color='#2c3e50', weight='bold')
        plt.text(i + width/2, dl_val + 0.005, f'{dl_val:.1%}', 
                ha='center', fontsize=7, color='#2c3e50', weight='bold')
    
    plt.xlabel('Adversarial Attack', fontsize=12, weight='bold')
    plt.ylabel('Recall (Detection Rate)', fontsize=12, weight='bold')
    plt.title('Robustezza agli Attacchi Adversarial: RF vs DL', fontsize=14, weight='bold')
    plt.xticks(x, attack_names, rotation=45, ha='right', fontsize=9)
    plt.ylim([0.0, 1.05])  # Resetto limite per vedere meglio eventuali crolli forti
    plt.legend(fontsize=10, loc='lower left')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'adversarial_robustness.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Grafico salvato: {save_path}")
    
    # --- CONCLUSIONE ---
    print("\n" + "="*70)
    print("   CONCLUSIONE SCIENTIFICA")
    print("="*70)
    
    if rf_avg_drop < dl_avg_drop:
        diff_pct = (dl_avg_drop - rf_avg_drop) / rf_avg_drop * 100
        print(f"\n[+] RANDOM FOREST e PIU ROBUSTO del Deep Learning")
        print(f"   -> RF mantiene {diff_pct:.1f}% piu recall sotto attacco")
        print(f"\n[INFO] Possibile spiegazione:")
        print(f"   - Random Forest: Ensemble di alberi -> decisione distribuita")
        print(f"   - Deep Learning: Gradiente continuo -> piu sensibile a perturbazioni")
    else:
        diff_pct = (rf_avg_drop - dl_avg_drop) / dl_avg_drop * 100
        print(f"\n[+] DEEP LEARNING e PIU ROBUSTO del Random Forest")
        print(f"   -> DL mantiene {diff_pct:.1f}% piu recall sotto attacco")
        print(f"\n[INFO] Possibile spiegazione:")
        print(f"   - Deep Learning: Impara features robuste attraverso layer multipli")
        print(f"   - Random Forest: Decision boundary troppo sensibile a valori esatti")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()