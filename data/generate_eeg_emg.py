"""
Synthetic EEG/EMG generator for 4-class motor intent:
Left Hand / Right Hand / Foot / Rest
Channels: C3, C4, Cz (motor cortex), EMG_L, EMG_R
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

FS = 256        # sampling rate Hz
DURATION = 1.0  # seconds per trial
N_TRIALS = 200  # per class
CLASSES = {0: "Left Hand", 1: "Right Hand", 2: "Foot", 3: "Rest"}
CHANNELS = ["C3", "C4", "Cz", "EMG_L", "EMG_R"]

def generate_trial(intent_class, fs=FS, duration=DURATION):
    t = np.linspace(0, duration, int(fs * duration))
    n = len(t)
    signals = {}

    noise = lambda scale=1.0: np.random.randn(n) * scale

    if intent_class == 0:  # Left Hand — C3 alpha suppression (ERD)
        signals["C3"]   = np.sin(2*np.pi*10*t) * 0.3 + noise(0.5)   # suppressed alpha
        signals["C4"]   = np.sin(2*np.pi*10*t) * 1.5 + noise(0.4)   # normal alpha
        signals["Cz"]   = np.sin(2*np.pi*12*t) * 0.8 + noise(0.5)
        signals["EMG_L"]= np.sin(2*np.pi*60*t) * 2.0 + noise(0.3)   # high EMG
        signals["EMG_R"]= noise(0.3)

    elif intent_class == 1:  # Right Hand — C4 alpha suppression
        signals["C3"]   = np.sin(2*np.pi*10*t) * 1.5 + noise(0.4)
        signals["C4"]   = np.sin(2*np.pi*10*t) * 0.3 + noise(0.5)   # suppressed
        signals["Cz"]   = np.sin(2*np.pi*12*t) * 0.8 + noise(0.5)
        signals["EMG_L"]= noise(0.3)
        signals["EMG_R"]= np.sin(2*np.pi*60*t) * 2.0 + noise(0.3)

    elif intent_class == 2:  # Foot — Cz activity
        signals["C3"]   = np.sin(2*np.pi*10*t) * 1.0 + noise(0.4)
        signals["C4"]   = np.sin(2*np.pi*10*t) * 1.0 + noise(0.4)
        signals["Cz"]   = np.sin(2*np.pi*8*t)  * 2.5 + noise(0.3)   # strong Cz
        signals["EMG_L"]= np.sin(2*np.pi*40*t) * 0.8 + noise(0.3)
        signals["EMG_R"]= np.sin(2*np.pi*40*t) * 0.8 + noise(0.3)

    else:  # Rest — all channels low, alpha dominant everywhere
        signals["C3"]   = np.sin(2*np.pi*10*t) * 1.8 + noise(0.3)
        signals["C4"]   = np.sin(2*np.pi*10*t) * 1.8 + noise(0.3)
        signals["Cz"]   = np.sin(2*np.pi*10*t) * 1.5 + noise(0.3)
        signals["EMG_L"]= noise(0.2)
        signals["EMG_R"]= noise(0.2)

    return signals

def generate_dataset():
    rows = []
    for cls in range(4):
        for trial in range(N_TRIALS):
            sigs = generate_trial(cls)
            row = {"label": cls, "class_name": CLASSES[cls]}
            for ch, sig in sigs.items():
                row[f"{ch}_mean"]  = np.mean(sig)
                row[f"{ch}_std"]   = np.std(sig)
                row[f"{ch}_power"] = np.mean(sig**2)
                row[f"{ch}_max"]   = np.max(np.abs(sig))
                # Band power approximation
                fft = np.abs(np.fft.rfft(sig))
                freqs = np.fft.rfftfreq(len(sig), 1/FS)
                row[f"{ch}_alpha"] = np.mean(fft[(freqs>=8)&(freqs<=13)])
                row[f"{ch}_beta"]  = np.mean(fft[(freqs>=13)&(freqs<=30)])
                row[f"{ch}_gamma"] = np.mean(fft[(freqs>=30)&(freqs<=50)])
            rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "eeg_emg_dataset.csv")
    df.to_csv(out, index=False)
    print(f"Dataset saved: {out} — {len(df)} trials, {len(df.columns)-2} features")
    return df

if __name__ == "__main__":
    generate_dataset()
