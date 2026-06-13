"""
Leaky Integrate-and-Fire (LIF) spike encoder.
Converts continuous EEG/EMG signals into spike trains.
"""
import numpy as np

class LIFEncoder:
    def __init__(self, threshold=2.0, leak=0.9, dt=1/256):
        self.threshold = threshold
        self.leak = leak
        self.dt = dt

    def encode_signal(self, signal):
        """Convert a 1D signal into spike train (binary array)."""
        membrane = 0.0
        spikes = []
        for s in signal:
            membrane = self.leak * membrane + s * self.dt * 50
            if membrane >= self.threshold:
                spikes.append(1)
                membrane = 0.0
            else:
                spikes.append(0)
        return np.array(spikes)

    def encode_trial(self, signals_dict):
        """Encode all channels; return spike rate features per channel."""
        features = {}
        for ch, sig in signals_dict.items():
            spikes = self.encode_signal(sig)
            features[f"{ch}_spike_rate"] = spikes.mean()
            features[f"{ch}_spike_count"] = spikes.sum()
            # Interspike interval stats
            spike_idx = np.where(spikes)[0]
            if len(spike_idx) > 1:
                isi = np.diff(spike_idx)
                features[f"{ch}_isi_mean"] = isi.mean()
                features[f"{ch}_isi_std"]  = isi.std()
            else:
                features[f"{ch}_isi_mean"] = 0
                features[f"{ch}_isi_std"]  = 0
        return features
