import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import warnings

# Silenzia warning non critici
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Seed per riproducibilità
np.random.seed(42)
torch.manual_seed(42)

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT_FILE = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")
DL_MODEL = os.path.join(BASE_DIR, "models", "mlp_model.pth")
DL_SCALER = os.path.join(BASE_DIR, "models", "scaler_dl.pkl")

# --- DEFINIZIONE RETE NEURALE ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def pgd_attack(model, X, y, epsilon=0.5, alpha=0.02, num_iter=50):
    """
    Projected Gradient Descent (PGD)
    1. Calcola il gradiente dell'errore rispetto all'input.
    2. Modifica l'input per massimizzare l'errore (nascondere il malware).
    """
    # Inizializzazione casuale (Random Start) per evitare minimi locali
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    
    for t in range(num_iter):
        # Forward pass
        outputs = model(X + delta)
        
        # Calcolo Loss (Vogliamo che il modello sbagli, quindi massimizziamo la loss rispetto alla label vera '1')
        loss = nn.BCELoss()(outputs, y)
        
        # Calcolo Gradiente (Direzione di massima pendenza dell'errore)
        loss.backward()
        
        # Aggiornamento Input (Gradient Ascent)
        with torch.no_grad():
            delta.add_(alpha * delta.grad.sign())
            
            # Proiezione (Clamp): Mantiene le modifiche "realistiche" entro epsilon
            delta.clamp_(-epsilon, epsilon)
            
            # Reset gradiente
            delta.grad.zero_()
            
    return (X + delta).detach()

def run_attack():
    print("--- PGD WHITE-BOX ATTACK (Solo Deep Learning) ---")
    
    # 1. Carica e Filtra Dati (Solo Malware)
    df = pd.read_csv(HOLDOUT_FILE)
    df_mal = df[df['Label'] == 1].copy()
    
    # Preparazione tensori
    X_raw = df_mal.drop(columns=['Label']).values
    y_true = torch.ones(len(X_raw), 1) # Sono tutti malware (1)
    
    print(f"Target: {len(X_raw)} campioni Malware.")

    # 2. Carica Modello DL
    scaler = joblib.load(DL_SCALER)
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    model = SimpleMLP(input_dim=X_raw.shape[1])
    model.load_state_dict(torch.load(DL_MODEL, weights_only=True))
    model.eval() 
    
    # 3. Misura Recall Iniziale (Prima dell'attacco)
    with torch.no_grad():
        preds_orig = (model(X_tensor) > 0.5).float().numpy()
    recall_orig = preds_orig.sum() / len(preds_orig)
    print(f"Recall Iniziale: {recall_orig:.4f} (Il modello ne vede {int(preds_orig.sum())})")

    # 4. Esegui l'Attacco PGD
    print("\n--- Avvio Attacco PGD (Epsilon=0.5, Iter=50) ---")
    X_adv = pgd_attack(model, X_tensor, y_true, epsilon=0.5, alpha=0.02, num_iter=50)
    
    # 5. Valuta Recall dopo l'attacco
    with torch.no_grad():
        preds_adv = (model(X_adv) > 0.5).float().numpy()
    recall_adv = preds_adv.sum() / len(preds_adv)
    
    print(f"Recall Sotto Attacco: {recall_adv:.4f} (Il modello ne vede {int(preds_adv.sum())})")
    
    # 6. Analisi Finale
    drop = recall_orig - recall_adv
    print("\n--- RISULTATO ---")
    if recall_adv < 0.1:
        print(f"✅ SUCCESSO TOTALE: La rete è stata accecata (Recall crollata di -{drop:.2%}).")
        print("   L'attacco matematico ha reso i malware invisibili.")
    else:
        print(f"⚠️ PARZIALE: La rete resiste ancora un po' (Recall crollata di -{drop:.2%}).")

if __name__ == "__main__":
    run_attack()