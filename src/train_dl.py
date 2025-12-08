import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib

# --- CONFIGURAZIONE e IMPORTAZIONE PERCORSI FILE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(BASE_DIR, "data", "dev_set.csv") #90% dei dati per training + validation
TEST_FILE = os.path.join(BASE_DIR, "data", "holdout_dataset.csv") #10% dei dati per test finale
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR = os.path.join(BASE_DIR, "results", "plots") # Cartella per i grafici

MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_dl.pkl")

# Parametri Training
BATCH_SIZE = 128 #batch di dati  di dim standard per ogni iterazione
LEARNING_RATE = 0.001 #velocità con cui il modello aggiorna i pesi. (Adam funziona bene con 0.001)
EPOCHS = 30 #N volte in cui modello può vedere l'intero dataset
# Dato che rete è piccola e dataset tabellare, 30 bastano per far emergere overfitting.
PATIENCE = 7 #se per 7 epoche consecutive la validazione peggiora → stop

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- ARCHITETTURA RETE NEURALE ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64) #64 neuroni nel 1 strato
        self.fc2 = nn.Linear(64, 32) #32 neuroni nel 2 strato
        self.fc3 = nn.Linear(32, 1) #1 neurone di output (binary classification)
        self.relu = nn.ReLU() #ReLU(x)=max(0,x)
        self.dropout = nn.Dropout(0.2) #Dropout per regolarizzazione
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

def train_deep_learning():
    print("--- DEEP LEARNING: TRAINING MLP ---")
    
    # Check Cartelle
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
    
    # --- 1. CARICAMENTO DATI ----
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print("Errore: File dati mancanti. Esegui prima train_model.py")
        return

    print("Caricamento dataset...")
    df_dev = pd.read_csv(TRAIN_FILE)
    df_holdout = pd.read_csv(TEST_FILE)
    
    # --- 2. PREPARAZIONE DATI ---
    X_dev = df_dev.drop(columns=['Label']).values
    y_dev = df_dev['Label'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.2, random_state=42, stratify=y_dev
    )
    
    X_holdout = df_holdout.drop(columns=['Label']).values
    y_holdout = df_holdout['Label'].values
    
    # --- 3. NORMALIZZAZIONE ---
    print("Normalizzazione dati...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_holdout = scaler.transform(X_holdout)
    
    joblib.dump(scaler, SCALER_PATH)

    # Conversione in Tensori
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_holdout_t = torch.tensor(X_holdout, dtype=torch.float32)

    # --- 4. BILANCIAMENTO CLASSI ---
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[int(label)] for label in y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # --- 5. INIZIALIZZAZIONE MODELLO ---
    input_dim = X_train.shape[1]
    model = SimpleMLP(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 6. TRAINING LOOP ---
    print(f"\nAvvio Training ({EPOCHS} epoche)...")
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validazione
        model.eval()
        with torch.no_grad():
            X_val_d, y_val_d = X_val_t.to(device), y_val_t.to(device)
            val_pred = model(X_val_d)
            val_loss = criterion(val_pred, y_val_d).item()
            val_acc = accuracy_score(y_val, (val_pred > 0.5).float().cpu())
            
        print(f"Epoca {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping attivato.")
                break

    training_time = time.time() - start_time
    print(f"\nTempo Training: {training_time:.1f} sec")

    # --- 7. TEST FINALE SU HOLDOUT ---
    print("\n--- Valutazione Finale (Holdout Set) ---")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    with torch.no_grad():
        X_holdout_d = X_holdout_t.to(device)
        y_pred_prob = model(X_holdout_d)
        y_pred_cls = (y_pred_prob > 0.5).float().cpu().numpy().flatten()
        
    acc = accuracy_score(y_holdout, y_pred_cls)
    print(f"Accuratezza DL: {acc:.4f}")
    
    print("\nReport Dettagliato:")
    print(classification_report(y_holdout, y_pred_cls, target_names=['Benign', 'Malicious'], digits=4))
    
    # --- 8. MATRICE DI CONFUSIONE (Stampa + Grafico) ---
    cm = confusion_matrix(y_holdout, y_pred_cls)
    
    print("Matrice di Confusione (Numerica):")
    print(cm)
    
    # Stampa dettagliata
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()


    # Generazione Grafico
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'],
                annot_kws={"size": 14})
    
    plt.ylabel('Realtà')
    plt.xlabel('Predizione')
    plt.title('Matrice di Confusione - Deep Learning (MLP)')
    
    save_path = os.path.join(IMG_DIR, "confusion_matrix_dl.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Grafico salvato in: {save_path}")

    print(f"\n✅ Modello salvato in: {MODEL_PATH}")

if __name__ == "__main__":
    train_deep_learning()