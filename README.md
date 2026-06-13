# NeuroMotion — Stroke Motor Intent Detection

> Real-time motor intent detection from EEG/EMG signals using Spiking Neural Networks, enabling stroke patients to control assistive devices through thought alone.

---

## 🧠 What It Does

Stroke patients often retain motor intent (the *will* to move) even when physical movement is impaired. **NeuroMotion** detects that intent from brainwave (EEG) and muscle (EMG) signals, then triggers assistive devices in real-time — robotic exoskeletons, FES stimulators — to restore functional movement.

**4 classes detected:** Left Hand · Right Hand · Foot · Rest

---

## 🏗 Pipeline

```
EEG/EMG Headset (256 Hz)
        ↓
  LIF Spike Encoding          ← biologically plausible preprocessing
        ↓
  Feature Extraction           ← 35 features: band power, spike rates, ISI
        ↓
  SNN / Gradient Boosting      ← dual model: accuracy vs latency tradeoff
        ↓
  Intent Classification
        ↓
  Assistive Device Trigger     ← exoskeleton, FES stimulator
```

---

## 🏆 Key Technical Features

| Feature | Detail |
|---|---|
| **SNN (Spiking Neural Net)** | snntorch LIF neurons — neuromorphic-hardware ready, biologically plausible |
| **LIF Encoder** | Leaky Integrate-and-Fire spike encoding on raw EEG/EMG |
| **Dual Model** | GBM (0.76ms latency) vs SNN (9ms, more bioplausible) — user-selectable |
| **Explainable AI** | Channel-level contribution (C3/C4/Cz/EMG) per prediction |
| **Virtual Hand** | SVG animated hand responds to each detected intent |
| **Recovery Dashboard** | Per-patient session history, accuracy trend, trigger count |
| **False Alarm Cancellation** | Low-confidence predictions blocked before device trigger |
| **3 Patient Profiles** | Ischemic stroke, hemorrhagic stroke, TIA — different rehab stages |

---

## 🚀 Running the App

```bash
# 1. Install dependencies
pip install flask scikit-learn pandas numpy snntorch torch

# 2. Train models (first time only — takes ~2 min)
python model/train_classifier.py

# 3. Start dashboard
python app/app.py

# 4. Open in browser
http://127.0.0.1:5000
```

---

## 📊 Results (Synthetic Data)

| Model | Accuracy | Inference Latency | Notes |
|---|---|---|---|
| Gradient Boosting | 100% | 0.76ms | Fast, production-ready |
| Spiking Neural Net | 100% | 9.2ms | Biologically plausible, neuromorphic-suitable |

*100% accuracy expected on synthetic data — signals are cleanly separated by design for prototype validation. Real EEG would achieve 70-85% with proper preprocessing.*

---

## 🎯 Judging Talking Points

1. **Why SNN?** *"Spiking Neural Networks directly mimic biological neurons, making them ideal for neuromorphic hardware like Intel Loihi — orders of magnitude more power-efficient than conventional deep learning for always-on medical devices."*

2. **Why LIF encoding?** *"The brain communicates through spikes, not continuous voltages. LIF encoding converts raw EEG into biologically realistic spike trains before classification — the same principle used in actual neural interfaces."*

3. **Why this matters?** *"30 million stroke survivors worldwide lose motor function. BCIs that decode motor intent can give them back the ability to reach for a glass of water, wave to their children, or type a message — even when nerves are damaged."*

4. **Latency:** *"Our GBM path achieves sub-millisecond classification. With a 1-second window, total system latency is ~1 second — within the 2-3 second therapeutic window for FES-assisted motor relearning."*

5. **False alarm cancellation:** *"We threshold at 75% confidence before triggering any device. In medical AI, a false positive that fires a muscle stimulator unpredictably is worse than a missed detection."*

---

## 📁 File Structure

```
stroke-motor-intent/
├── data/
│   └── generate_eeg_emg.py      # Synthetic 4-class EEG/EMG generator
├── encoding/
│   └── lif_encoder.py            # LIF neuron spike encoder
├── model/
│   ├── train_classifier.py       # SNN + GBM training
│   ├── gbm_model.pkl             # Trained GBM (auto-generated)
│   ├── snn_model.pt              # Trained SNN (auto-generated)
│   ├── scaler.pkl                # Feature scaler (auto-generated)
│   └── meta.json                 # Model metadata (auto-generated)
├── app/
│   ├── app.py                    # Flask backend
│   └── templates/
│       └── dashboard.html        # Full dashboard UI
└── README.md
```
Team:
Leader : Tanisha Dangwal
Contribution: Vaibhav Murti
