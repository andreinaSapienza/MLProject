# IoT Cryptomining Detection System

## 🎯 Obiettivo
Sistema di rilevamento automatico di attacchi cryptomining (Webmine, Binary payload) su dispositivi IoT Raspberry Pi attraverso analisi del traffico di rete con Machine Learning.

## 📊 Risultati Principali

### Performance su Holdout Set (62,100 campioni mai visti)
```
Accuracy:          99%
Recall Malicious:  92%  (4,972/5,400 attacchi rilevati)
Precision:         92%
False Positive:    0.7% (411 falsi allarmi su 56,700 campioni benign)
Gap Val-Holdout:   <2%  (modello robusto, zero overfitting)
```

### Matrice di Confusione
```
                 Predicted
                 Benign  Malicious
Actual Benign    56,289     411
       Malicious    428   4,972
```

## 🏗️ Architettura del Progetto

```
LabML/
├── data/
│   ├── benign/              # Traffico normale (video, idle)
│   ├── malicious/           # Attacchi (Webmine, Binary)
│   ├── processed_traffic.csv       # Pacchetti filtrati per MAC
│   ├── final_dataset_windowed.csv  # Features estratte (rolling window)
│   ├── dev_set.csv                 # Development set (90%)
│   └── holdout_dataset.csv         # Holdout test set (10%, mai visto)
├── src/
│   ├── main.py              # Pre-processing e filtraggio MAC
│   ├── build_features.py    # Feature engineering (windowing)
│   ├── train_model.py       # Training con holdout split
│   └── predict.py           # Inference su holdout set
├── models/
│   ├── rf_model.pkl         # Random Forest addestrato
│   └── scaler.pkl           # StandardScaler per normalizzazione
└── utils/
    └── loader.py            # Funzioni per caricamento dati
```

## 🔬 Metodologia

### 1. Pre-Processing
- **Input:** File CSV con traffico di rete (Timestamp, Length, MAC addresses)
- **Filtraggio:** Selezione pacchetti del Raspberry Pi target (Src o Dst MAC)
- **Output:** `processed_traffic.csv` (620,205 pacchetti)

### 2. Feature Engineering
- **Tecnica:** Rolling Window (10 pacchetti consecutivi)
- **Features Temporali:** 
  - `Time_Mean`: Media tempo tra pacchetti
  - `Time_Var`: Varianza tempo (rileva pattern periodici del mining)
- **Features Dimensionali:**
  - `Length_Mean`, `Length_Min`, `Length_Max`: Statistiche dimensione pacchetti
  - `Length_Var`: Varianza dimensione
- **Output:** `final_dataset_windowed.csv` (620,205 finestre)

### 3. Training
- **Algoritmo:** Random Forest (100 trees, class_weight='balanced')
- **Split:** 
  - Holdout Set (10%, 62,100 campioni) separato **prima** del training
  - Dev Set (90%, 558,105 campioni) diviso in train/validation (80/20)
- **Tecnica Split:** Block-based (gruppi da 100) per rispettare rolling windows
- **Normalizzazione:** StandardScaler (fit su training, transform su val/holdout)

### 4. Validazione
- **Test Finale:** Solo su holdout set mai visto dal modello
- **Metriche:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Analisi:** Confidenza predizioni, distribuzione errori

## 🚀 Quick Start

### Requisiti
```bash
Python 3.8+
pandas
numpy
scikit-learn
joblib
```

### Installazione
```bash
# Clona il repository
git clone <repository-url>
cd LabML

# Crea ambiente virtuale
python3 -m venv venv
source venv/bin/activate

# Installa dipendenze
pip install pandas numpy scikit-learn joblib
```

### Esecuzione Pipeline Completo

#### 1. Pre-processing
```bash
python3 src/main.py
# Output: data/processed_traffic.csv
```

#### 2. Feature Engineering
```bash
python3 src/build_features.py
# Output: data/final_dataset_windowed.csv
```

#### 3. Training
```bash
python3 src/train_model.py
# Output: 
#   - models/rf_model.pkl
#   - models/scaler.pkl
#   - data/holdout_dataset.csv (creato automaticamente)
```

#### 4. Test su Holdout
```bash
python3 src/predict.py
# Mostra: Performance su holdout set + 20 predizioni casuali
```

## 📈 Risultati Dettagliati

### Validation Set (111,700 campioni)
```
              precision    recall  f1-score   support
Benign            0.99      0.99      0.99     99,200
Malicious         0.94      0.93      0.93     12,500
accuracy                             0.99    111,700
```

### Holdout Set (62,100 campioni - MAI VISTO)
```
              precision    recall  f1-score   support
Benign            0.99      0.99      0.99     56,700
Malicious         0.92      0.92      0.92      5,400
accuracy                             0.99     62,100
```

### Feature Importance (Top 3)
1. **Time_Mean** (36.7%): Tempo medio tra pacchetti - distingue mining (costante) da video (variabile)
2. **Length_Max** (25.1%): Dimensione massima pacchetto
3. **Length_Mean** (18.2%): Dimensione media pacchetto

## 🔍 Punti di Forza

### 1. Metodologia Rigorosa
- ✅ **Holdout set separato** prima del training (evita data leakage)
- ✅ **Split temporale** (rispetta ordine dei dati time-series)
- ✅ **Block-based split** (compatibile con rolling windows)

### 2. Risultati Robusti
- ✅ **Gap validation-holdout <2%** (zero overfitting)
- ✅ **Confidenza realistica** (non tutti i campioni a 100%)
- ✅ **Consistenza temporale** (ultimi 10% del dataset come test)

### 3. Competitività
- ✅ **Recall 92%** allineato con paper scientifici (range 89-94%)
- ✅ **False Positive Rate 0.7%** (eccellente per sistema di alerting)
- ✅ **99% accuracy** su dati mai visti

## ⚠️ Limitazioni e Lavori Futuri

### Limitazioni Attuali
1. **Recall 92%**: 8% di attacchi non rilevati
   - 428 falsi negativi su 5,400 attacchi
   - Critico per sistemi di sicurezza real-time
2. **Dataset specifico**: Testato solo su Raspberry Pi del paper
3. **Feature limitate**: Solo statistiche temporali/dimensionali

### Miglioramenti Proposti
1. **Threshold Tuning**: Abbassare soglia decisionale per recall 95%+
2. **Feature Engineering**: 
   - Burst detection (pacchetti ravvicinati)
   - Periodicità (pattern ciclici del mining)
   - Rapporti dimensionali (max/min)
3. **Ensemble Methods**: Combinare RF con XGBoost/SVM
4. **Test Esterni**: Validare su CICIoT2023 o N-BaIoT datasets

## 📚 Dataset Originale

Basato sul paper:
- **Titolo:** IoT Cryptomining Detection  
- **Device:** Raspberry Pi 3B+
- **Attacchi:** 
  - Webmine (browser-based cryptomining)
  - Binary payload (executable malware)
- **Traffico Benign:** Video streaming, Idle traffic

## 👤 Autore

Progetto di Machine Learning - Sapienza Università di Roma  
Anno Accademico: 2024/2025

## 📄 Licenza

Progetto accademico - Uso educativo

## 🙏 Ringraziamenti

- Dataset: Paper originale su IoT Security
- Supervisione: Prof. [Nome Docente]
- Tools: scikit-learn, pandas

---

**Note:** Questo progetto dimostra l'importanza di una validazione rigorosa in Machine Learning, 
evitando data leakage attraverso holdout set separato e rispettando la natura time-series dei dati.
