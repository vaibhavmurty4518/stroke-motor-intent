<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=NeuroMotion&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Stroke%20Motor%20Intent%20Detection%20via%20SNN&descAlignY=60&descSize=20" width="100%"/>

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Railway-8A2BE2?style=for-the-badge&logo=railway&logoColor=white)](https://web-production-6556e.up.railway.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-SNN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **Real-time motor intent detection from EEG/EMG signals using Spiking Neural Networks —**  
> **enabling stroke patients to control assistive devices through thought alone.**

<br/>

```
30 million stroke survivors worldwide  •  Motor intent preserved even when movement isn't
```

</div>

---

## 🧠 What Is NeuroMotion?

Stroke patients often retain **motor intent** — the will to move — even when physical movement is severely impaired. **NeuroMotion** detects that intent from brainwave (EEG) and muscle (EMG) signals, then triggers assistive devices in real-time:

- 🦾 **Robotic Exoskeletons** — to physically assist movement
- ⚡ **FES Stimulators** — Functional Electrical Stimulation for motor relearning

### 4 Motor Classes Detected

| 🤚 Left Hand | ✋ Right Hand | 🦶 Foot | 😴 Rest |
|:---:|:---:|:---:|:---:|
| Left motor cortex activation | Right motor cortex activation | Lower limb intent | Baseline / no intent |

---

## 🏗️ System Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   EEG/EMG Headset (256 Hz)                                  │
│          │                                                  │
│          ▼                                                  │
│   ⚡ LIF Spike Encoding      ← biologically plausible       │
│          │                                                  │
│          ▼                                                  │
│   📊 Feature Extraction      ← 35 features: band power,     │
│          │                      spike rates, ISI            │
│          ▼                                                  │
│   🧬 SNN / Gradient Boosting ← dual model: accuracy vs      │
│          │                      latency tradeoff            │
│          ▼                                                  │
│   🎯 Intent Classification                                  │
│          │                                                  │
│          ▼                                                  │
│   🦾 Assistive Device Trigger ← exoskeleton / FES          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏆 Key Technical Features

<table>
<tr>
<td width="50%">

### 🧬 Spiking Neural Network
- **snntorch** Leaky Integrate-and-Fire (LIF) neurons
- Neuromorphic-hardware ready (Intel Loihi compatible)
- Biologically plausible computation model
- Mimics real neural spike communication

</td>
<td width="50%">

### ⚡ LIF Spike Encoder
- Converts raw EEG/EMG → spike trains
- Same principle used in real neural interfaces
- Brain communicates in spikes, not voltages
- Biologically faithful preprocessing

</td>
</tr>
<tr>
<td>

### 🔀 Dual Model Architecture
| Model | Latency | Use Case |
|-------|---------|----------|
| **GBM** | 0.76ms | Production speed |
| **SNN** | 9ms | Bioplausible |

User-selectable based on clinical need

</td>
<td>

### 🔍 Explainable AI (XAI)
- Per-prediction **channel contribution**
- Tracks: **C3 / C4 / Cz / EMG** channels
- Clinician-interpretable outputs
- Trust-first medical AI design

</td>
</tr>
<tr>
<td>

### 🛡️ False Alarm Cancellation
- **75% confidence threshold** before device trigger
- In medical AI: false positive = unpredicted muscle stimulation
- Patient safety is non-negotiable
- Blocked low-confidence predictions never reach hardware

</td>
<td>

### 📈 Recovery Dashboard
- Per-patient **session history**
- Accuracy trend visualization
- Trigger count tracking
- 3 Patient Profiles: Ischemic · Hemorrhagic · TIA

</td>
</tr>
</table>

---

## 📊 Model Performance

<div align="center">

| Model | Accuracy | Inference Latency | Notes |
|:---:|:---:|:---:|:---|
| 🚀 **Gradient Boosting** | `100%` | `0.76 ms` | Fast, production-ready |
| 🧠 **Spiking Neural Net** | `100%` | `9.2 ms` | Biologically plausible, neuromorphic-suitable |

> ⚠️ **Note:** 100% accuracy is expected on synthetic data — signals are cleanly separated by design for prototype validation.  
> Real EEG data would yield **70–85% accuracy** with proper preprocessing pipelines.

</div>

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/vaibhavmurty4518/stroke-motor-intent.git
cd stroke-motor-intent

# 2. Install dependencies
pip install flask scikit-learn pandas numpy snntorch torch

# 3. Train models (first time only — ~2 minutes)
python model/train_classifier.py

# 4. Launch the dashboard
python app/app.py

# 5. Open in browser
open http://127.0.0.1:5000
```

Or access the **live deployment** instantly:  
🌐 **[https://web-production-6556e.up.railway.app/](https://web-production-6556e.up.railway.app/)**

---

## 📁 Project Structure

```
stroke-motor-intent/
│
├── 📂 data/
│   └── generate_eeg_emg.py        # Synthetic 4-class EEG/EMG generator
│
├── 📂 encoding/
│   └── lif_encoder.py             # LIF neuron spike encoder
│
├── 📂 model/
│   ├── train_classifier.py        # SNN + GBM training pipeline
│   ├── gbm_model.pkl              # Trained GBM (auto-generated)
│   ├── snn_model.pt               # Trained SNN (auto-generated)
│   ├── scaler.pkl                 # Feature scaler (auto-generated)
│   └── meta.json                  # Model metadata (auto-generated)
│
├── 📂 app/
│   ├── app.py                     # Flask backend
│   └── templates/
│       └── dashboard.html         # Full recovery dashboard UI
│
├── Dockerfile                     # Container deployment
├── Procfile                       # Railway process config
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python version pin
└── README.md
```

---

## 🎯 Why This Matters — Talking Points

<details>
<summary><b>🧠 Why Spiking Neural Networks?</b></summary>

> Spiking Neural Networks directly mimic biological neurons, making them ideal for neuromorphic hardware like **Intel Loihi** — orders of magnitude more power-efficient than conventional deep learning for always-on medical devices.

</details>

<details>
<summary><b>⚡ Why LIF Encoding?</b></summary>

> The brain communicates through spikes, not continuous voltages. LIF encoding converts raw EEG into biologically realistic spike trains before classification — the same principle used in actual neural interfaces.

</details>

<details>
<summary><b>💙 Why This Matters?</b></summary>

> **30 million stroke survivors worldwide** lose motor function. BCIs that decode motor intent can give them back the ability to reach for a glass of water, wave to their children, or type a message — even when nerves are damaged.

</details>

<details>
<summary><b>⏱️ Latency Considerations</b></summary>

> Our GBM path achieves **sub-millisecond classification**. With a 1-second signal window, total system latency is ~1 second — within the **2–3 second therapeutic window** for FES-assisted motor relearning.

</details>

<details>
<summary><b>🛡️ False Alarm Cancellation</b></summary>

> We threshold at **75% confidence** before triggering any device. In medical AI, a false positive that fires a muscle stimulator unpredictably is worse than a missed detection.

</details>

---

## 👥 Team

<div align="center">

<table>
<tr>
<td align="center" width="50%">

### 👑 Team Leader
**Tanisha Dangwal**  
*BCI Architecture · SNN Design · Clinical Strategy*

</td>
<td align="center" width="50%">

### 🤝 Collaborator
**Vaibhav Murty**  
*Backend · Deployment · Pipeline Integration*

</td>
</tr>
</table>

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**Built with 💜 for 30 million stroke survivors**  

*NeuroMotion — Giving motion back through the power of thought*

</div>
