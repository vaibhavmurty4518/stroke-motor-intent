"""
Dual-model training:
1. Spiking Neural Network (snntorch) — biologically plausible, neuromorphic-ready
2. Gradient Boosting — fast inference fallback
Both saved to disk for use by Flask app.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.generate_eeg_emg import generate_dataset, generate_trial, CLASSES

# ── Generate / load dataset ────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "eeg_emg_dataset.csv")
if not os.path.exists(data_path):
    df = generate_dataset()
else:
    df = pd.read_csv(data_path)

feature_cols = [c for c in df.columns if c not in ("label","class_name")]
X = df[feature_cols].values.astype(np.float32)
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

MODEL_DIR = os.path.dirname(__file__)

# ── 1. Gradient Boosting ───────────────────────────────────────────────────
print("Training GBM...")
gbm = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
t0 = time.time()
gbm.fit(X_train_s, y_train)
gbm_train_time = time.time() - t0

preds_gbm = gbm.predict(X_test_s)
gbm_acc = (preds_gbm == y_test).mean()
print(f"GBM Accuracy: {gbm_acc:.4f}")
print(classification_report(y_test, preds_gbm, target_names=list(CLASSES.values())))

# Inference latency
t0 = time.time()
for _ in range(100):
    gbm.predict(X_test_s[:1])
gbm_latency_ms = (time.time()-t0)/100*1000
print(f"GBM inference latency: {gbm_latency_ms:.2f}ms")

# Feature importance for explainability
importances = gbm.feature_importances_
feat_imp = dict(zip(feature_cols, importances.tolist()))
top_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]

# ── 2. SNN (snntorch) ──────────────────────────────────────────────────────
print("\nTraining SNN...")
try:
    import torch
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate

    INPUT_DIM  = X_train_s.shape[1]
    HIDDEN_DIM = 64
    OUTPUT_DIM = 4
    TIMESTEPS  = 20
    EPOCHS     = 30
    LR         = 5e-3

    spike_grad = surrogate.fast_sigmoid(slope=25)

    class SpikingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1   = nn.Linear(INPUT_DIM, HIDDEN_DIM)
            self.lif1  = snn.Leaky(beta=0.9, spike_grad=spike_grad)
            self.fc2   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.lif2  = snn.Leaky(beta=0.9, spike_grad=spike_grad)
            self.fc3   = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
            self.lif3  = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            spk3_rec = []
            for _ in range(TIMESTEPS):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                cur3 = self.fc3(spk2)
                spk3, mem3 = self.lif3(cur3, mem3)
                spk3_rec.append(spk3)
            return torch.stack(spk3_rec, dim=0)  # [T, batch, classes]

    net = SpikingNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train_s, dtype=torch.float32)
    yt = torch.tensor(y_train,   dtype=torch.long)
    Xv = torch.tensor(X_test_s,  dtype=torch.float32)
    yv = torch.tensor(y_test,    dtype=torch.long)

    BATCH = 64
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            xb = Xt[idx[i:i+BATCH]]
            yb = yt[idx[i:i+BATCH]]
            spk_out = net(xb)                        # [T, B, C]
            rate    = spk_out.mean(0)                # [B, C] — spike rate decoding
            loss    = loss_fn(rate, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (ep+1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                rate_v = net(Xv).mean(0)
                acc_v  = (rate_v.argmax(1) == yv).float().mean().item()
            print(f"  Epoch {ep+1}/{EPOCHS} — val acc: {acc_v:.4f}")

    snn_train_time = time.time() - t0

    net.eval()
    with torch.no_grad():
        rate_v = net(Xv).mean(0)
        snn_preds = rate_v.argmax(1).numpy()
        snn_acc   = (snn_preds == y_test).mean()
        # Spike count for reporting
        spk_out_v = net(Xv)
        avg_spikes = spk_out_v.sum(0).mean().item()

    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            net(Xv[:1])
    snn_latency_ms = (time.time()-t0)/100*1000

    print(f"SNN Accuracy: {snn_acc:.4f}, avg spikes/sample: {avg_spikes:.2f}, latency: {snn_latency_ms:.2f}ms")
    print(classification_report(y_test, snn_preds, target_names=list(CLASSES.values())))

    torch.save(net.state_dict(), os.path.join(MODEL_DIR, "snn_model.pt"))
    snn_available = True

except Exception as e:
    print(f"SNN training failed: {e}")
    snn_available = False
    snn_acc = 0.0
    snn_latency_ms = 0.0
    avg_spikes = 0.0

# ── Save GBM + scaler + metadata ──────────────────────────────────────────
joblib.dump(gbm,    os.path.join(MODEL_DIR, "gbm_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

meta = {
    "feature_cols": feature_cols,
    "classes": CLASSES,
    "gbm_accuracy": float(gbm_acc),
    "gbm_latency_ms": float(gbm_latency_ms),
    "snn_available": snn_available,
    "snn_accuracy": float(snn_acc),
    "snn_latency_ms": float(snn_latency_ms),
    "snn_avg_spikes": float(avg_spikes) if snn_available else 0,
    "top_features": top_feats,
    "input_dim": int(X_train_s.shape[1]),
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n✓ All models saved.")
print(f"  GBM  — acc: {gbm_acc:.4f}, latency: {gbm_latency_ms:.2f}ms")
if snn_available:
    print(f"  SNN  — acc: {snn_acc:.4f}, latency: {snn_latency_ms:.2f}ms, spikes: {avg_spikes:.2f}/sample")
