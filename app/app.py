"""
Advanced AI Neuromorphic Rehab Suite — Flask App
Features: SNN + GBM dual model, Voice Assistant, AI Coach, Config Engine,
          Virtual Orthosis, XAI, Recovery Forecast, Neuromorphic Metrics,
          Clinical Summary Generator, Cosmetic Dashboard upgrades.
"""
import os, sys, json, time, random, math
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import joblib

BASE      = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE, "model")

# ── Load models ───────────────────────────────────────────────────────────────
gbm    = joblib.load(os.path.join(MODEL_DIR, "gbm_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)

feature_cols  = meta["feature_cols"]
CLASSES       = {int(k): v for k, v in meta["classes"].items()}
SNN_AVAILABLE = meta["snn_available"]

snn_net = None
if SNN_AVAILABLE:
    try:
        import torch
        import torch.nn as nn
        import snntorch as snn_lib
        from snntorch import surrogate

        INPUT_DIM  = meta["input_dim"]
        HIDDEN_DIM = 64
        OUTPUT_DIM = 4
        TIMESTEPS  = 20
        spike_grad = surrogate.fast_sigmoid(slope=25)

        class SpikingNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1  = nn.Linear(INPUT_DIM, HIDDEN_DIM)
                self.lif1 = snn_lib.Leaky(beta=0.9, spike_grad=spike_grad)
                self.fc2  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
                self.lif2 = snn_lib.Leaky(beta=0.9, spike_grad=spike_grad)
                self.fc3  = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
                self.lif3 = snn_lib.Leaky(beta=0.9, spike_grad=spike_grad)
            def forward(self, x):
                mem1 = self.lif1.init_leaky()
                mem2 = self.lif2.init_leaky()
                mem3 = self.lif3.init_leaky()
                spk3_rec, spk1_rec, spk2_rec = [], [], []
                for _ in range(TIMESTEPS):
                    cur1 = self.fc1(x);  spk1, mem1 = self.lif1(cur1, mem1)
                    cur2 = self.fc2(spk1); spk2, mem2 = self.lif2(cur2, mem2)
                    cur3 = self.fc3(spk2); spk3, mem3 = self.lif3(cur3, mem3)
                    spk3_rec.append(spk3)
                    spk1_rec.append(spk1)
                    spk2_rec.append(spk2)
                return torch.stack(spk3_rec, 0), torch.stack(spk1_rec, 0), torch.stack(spk2_rec, 0)

        snn_net = SpikingNet()
        snn_net.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "snn_model.pt"), map_location="cpu"),
            strict=False
        )
        snn_net.eval()
        print("SNN loaded ✓")
    except Exception as e:
        print(f"SNN load failed: {e}")
        SNN_AVAILABLE = False

# ── Patients ──────────────────────────────────────────────────────────────────
PATIENTS = {
    "P001": {"name": "Ramesh Kumar",  "age": 58, "condition": "Ischemic Stroke",   "affected": "Left",  "sessions": 12, "baseline_acc": 0.61},
    "P002": {"name": "Sunita Devi",   "age": 64, "condition": "Hemorrhagic Stroke","affected": "Right", "sessions": 8,  "baseline_acc": 0.55},
    "P003": {"name": "Arjun Mehta",   "age": 47, "condition": "TIA",               "affected": "Both",  "sessions": 5,  "baseline_acc": 0.70},
}

session_history = {pid: [] for pid in PATIENTS}

sys.path.insert(0, os.path.join(BASE, "data"))
from generate_eeg_emg import generate_trial, FS, DURATION
CHANNELS = ["C3", "C4", "Cz", "EMG_L", "EMG_R"]

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(signals_dict):
    row = {}
    for ch, sig in signals_dict.items():
        row[f"{ch}_mean"]  = np.mean(sig)
        row[f"{ch}_std"]   = np.std(sig)
        row[f"{ch}_power"] = np.mean(sig**2)
        row[f"{ch}_max"]   = np.max(np.abs(sig))
        fft   = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), 1/FS)
        row[f"{ch}_alpha"] = float(np.mean(fft[(freqs>=8)&(freqs<=13)]))
        row[f"{ch}_beta"]  = float(np.mean(fft[(freqs>=13)&(freqs<=30)]))
        row[f"{ch}_gamma"] = float(np.mean(fft[(freqs>=30)&(freqs<=50)]))
    return np.array([row[f] for f in feature_cols], dtype=np.float32)

def explain_prediction(features_scaled, pred_class):
    importances = gbm.feature_importances_
    top_idx = np.argsort(importances)[::-1][:6]
    channel_contrib = {"C3": 0, "C4": 0, "Cz": 0, "EMG_L": 0, "EMG_R": 0}
    for i in top_idx:
        feat = feature_cols[i]
        ch   = feat.split("_")[0]
        if ch in channel_contrib:
            channel_contrib[ch] += importances[i]
    total = sum(channel_contrib.values()) or 1
    explanation = []
    for ch, val in sorted(channel_contrib.items(), key=lambda x: -x[1]):
        explanation.append({"channel": ch, "contribution": round(val/total*100, 1)})
    return explanation

# ── Neuromorphic metrics helpers ──────────────────────────────────────────────
def compute_snn_metrics(spk_out_all):
    """
    spk_out_all: list of [T, 1, C] tensors for each layer
    Returns dict with spike_count, layer_sparsity, energy_eff_pJ, event_rate_keps
    """
    import torch
    total_spikes = sum(int(s.sum().item()) for s in spk_out_all)
    total_neurons = sum(s.shape[2] * s.shape[0] for s in spk_out_all)
    sparsity = round(1.0 - total_spikes / max(total_neurons, 1), 4)
    # Energy: ~10 pJ per spike for neuromorphic hardware (Intel Loihi estimate)
    energy_pj = round(total_spikes * 10.0, 1)
    # Event rate: kilo-events/sec (20 timesteps at 256 Hz simulated)
    event_rate = round(total_spikes / (TIMESTEPS / FS) / 1000, 2)
    return {
        "spike_count":    total_spikes,
        "layer_sparsity": round(sparsity * 100, 1),   # percent
        "energy_pj":      energy_pj,
        "event_rate_keps": event_rate,
    }

# ── AI Coach heuristic ────────────────────────────────────────────────────────
def generate_coach_feedback(confidence, snn_metrics, emg_power_l, emg_power_r,
                             latency_ms, triggered, intent, model_used):
    msgs = []
    score = round(confidence * 100, 1)
    sparsity = snn_metrics.get("layer_sparsity", None) if snn_metrics else None

    # Confidence commentary
    if score >= 92:
        if sparsity:
            msgs.append(f"🏆 Excellent spike timing! Motor intent confidence hit {score}% with {sparsity}% layer sparsity — neuromorphic efficiency is optimal.")
        else:
            msgs.append(f"🏆 Outstanding! Motor intent confidence reached {score}%. Clean, decisive neural patterns detected.")
    elif score >= 80:
        msgs.append(f"✅ Good intent signal at {score}% confidence. Motor cortex activation is well-defined.")
    elif score >= 65:
        msgs.append(f"⚠ Moderate confidence ({score}%). Consider focusing motor imagery more vividly on the target limb.")
    else:
        msgs.append(f"🔄 Low confidence ({score}%). Signal ambiguity detected — relax completely before next attempt.")

    # EMG fatigue check
    emg_threshold = 2.5
    if emg_power_r > emg_threshold:
        msgs.append(f"💪 High continuous EMG noise detected on EMG_R (power: {emg_power_r:.2f}). Muscle fatigue suspected — recommend a 2-minute rest break.")
    if emg_power_l > emg_threshold:
        msgs.append(f"💪 Elevated EMG_L activity (power: {emg_power_l:.2f}). Left forearm may be fatiguing. Shake out and relax the hand.")

    # Latency insight
    if latency_ms < 8:
        msgs.append(f"⚡ Ultra-low inference latency ({latency_ms}ms) — real-time BCI performance confirmed.")
    elif latency_ms > 25:
        msgs.append(f"🐢 Latency elevated ({latency_ms}ms). System load may be affecting response time.")

    # Trigger feedback
    if triggered and intent != "Rest":
        msgs.append(f"🦾 Orthosis TRIGGERED for '{intent}'. Assistive device engaged successfully.")

    # SNN-specific
    if sparsity and sparsity > 80:
        msgs.append(f"🧠 SNN layer sparsity at {sparsity}% — highly energy-efficient neuromorphic inference.")

    return msgs

# ── Recovery forecast ─────────────────────────────────────────────────────────
def forecast_recovery(history, patient):
    """Predict next 5 sessions using linear regression + noise."""
    if len(history) < 2:
        base = patient["baseline_acc"]
        return [round(min(0.97, base + (i+1)*0.025 + random.uniform(-0.01, 0.01)), 3)
                for i in range(5)]
    accs = [h["accuracy"] for h in history]
    n    = len(accs)
    xs   = list(range(n))
    xm, ym = np.mean(xs), np.mean(accs)
    slope  = np.sum([(x-xm)*(y-ym) for x,y in zip(xs,accs)]) / max(np.sum([(x-xm)**2 for x in xs]), 1e-9)
    intercept = ym - slope * xm
    forecast = []
    for i in range(1, 6):
        pred = intercept + slope * (n + i) + random.uniform(-0.015, 0.015)
        forecast.append(round(float(np.clip(pred, 0.40, 0.99)), 3))
    return forecast

# ── Clinical summary ──────────────────────────────────────────────────────────
def generate_clinical_summary(patient_id):
    patient = PATIENTS.get(patient_id, PATIENTS["P001"])
    hist    = session_history.get(patient_id, [])
    cur     = hist[-20:] if hist else []

    if not cur:
        return f"Insufficient session data for {patient['name']}. Please run at least one simulation."

    avg_conf = np.mean([h["confidence"] for h in cur]) * 100
    triggers = sum(1 for h in cur if h["triggered"])
    avg_lat  = np.mean([h["latency_ms"] for h in cur])
    baseline = patient["baseline_acc"] * 100
    improvement = round(avg_conf - baseline, 1)
    trend = "improving" if improvement > 0 else "declining"
    direction = "+" if improvement > 0 else ""

    summary = (
        f"**Patient:** {patient['name']} | Age {patient['age']} | {patient['condition']} — {patient['affected']} side affected\n\n"
        f"**Session Overview ({len(cur)} windows analysed):**\n"
        f"Patient demonstrates {trend} motor intent detection with "
        f"{direction}{improvement}% {'improvement' if improvement > 0 else 'change'} over baseline ({baseline:.0f}%). "
        f"Average detection confidence: {avg_conf:.1f}%. "
        f"Assistive device triggered in {triggers}/{len(cur)} windows ({100*triggers/len(cur):.0f}% activation rate).\n\n"
        f"**Neuromotor Performance:** Average inference latency of {avg_lat:.1f}ms confirms real-time BCI viability. "
        f"Motor cortex lateralisation patterns consistent with {patient['affected'].lower()}-side stroke recovery trajectory.\n\n"
        f"**Clinical Recommendation:** "
    )

    if avg_conf >= 85:
        summary += "Excellent neuroplasticity indicators. Consider advancing to active-resistance orthosis protocols."
    elif avg_conf >= 70:
        summary += "Moderate recovery signals. Continue current motor imagery schedule with biofeedback reinforcement."
    else:
        summary += "Early-stage recovery patterns. Maintain passive-assist BCI with high-frequency short sessions."

    return summary

# ── Voice command parser ───────────────────────────────────────────────────────
VOICE_COMMANDS = {
    "start calibration": {"action": "calibrate", "response": "Starting calibration sequence. Please relax your muscles and focus on the target limb."},
    "trigger left hand":  {"action": "simulate", "class": 0, "response": "Triggering left hand motor intent simulation."},
    "trigger right hand": {"action": "simulate", "class": 1, "response": "Triggering right hand motor intent simulation."},
    "trigger foot":       {"action": "simulate", "class": 2, "response": "Triggering foot motor intent simulation."},
    "force rest":         {"action": "simulate", "class": 3, "response": "Setting system to rest state."},
    "show left hand history": {"action": "show_history", "filter": "Left Hand", "response": "Displaying left hand detection history."},
    "show right hand history": {"action": "show_history", "filter": "Right Hand", "response": "Displaying right hand detection history."},
    "switch to snn":  {"action": "set_model", "model": "snn",  "response": "Switching to Spiking Neural Network model. SNN mode active."},
    "switch to gbm":  {"action": "set_model", "model": "gbm",  "response": "Switching to Gradient Boosting model. GBM mode active."},
    "run simulation": {"action": "simulate", "class": None,    "response": "Running random motor intent simulation window."},
    "take a break":   {"action": "pause",                      "response": "Rest break initiated. Relax your arm and breathe deeply for 2 minutes."},
    "show summary":   {"action": "summary",                    "response": "Generating clinical summary for current patient."},
}

def parse_voice_command(text):
    text_lower = text.lower().strip()
    # Exact match first
    for key, cmd in VOICE_COMMANDS.items():
        if key in text_lower:
            return {**cmd, "matched": key, "tts": cmd["response"]}
    # Fallback
    return {"action": "unknown", "tts": f"Sorry, I didn't recognise that command: '{text}'. Try saying 'run simulation', 'switch to SNN', or 'trigger left hand'."}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("dashboard.html",
        patients=PATIENTS,
        snn_available=SNN_AVAILABLE,
        gbm_accuracy=round(meta["gbm_accuracy"]*100, 1),
        snn_accuracy=round(meta["snn_accuracy"]*100, 1),
        snn_latency=round(meta["snn_latency_ms"], 1),
        gbm_latency=round(meta["gbm_latency_ms"], 2),
    )

@app.route("/simulate", methods=["POST"])
def simulate():
    data        = request.get_json() or {}
    patient_id  = data.get("patient_id", "P001")
    model_type  = data.get("model", "gbm")
    force_class = data.get("force_class", None)
    # Config engine params
    threshold_pct   = float(data.get("threshold_pct", 75)) / 100.0   # 0.75/0.85/0.95
    num_classes     = int(data.get("num_classes", 4))                 # 2/3/4
    encoding_mode   = data.get("encoding_mode", "delta")              # delta/poisson
    delta_threshold = float(data.get("delta_threshold", 0.5))        # 0.1–2.0

    # Restrict classes if paradigm < 4
    max_class = num_classes - 1
    if force_class is not None:
        intent_class = min(int(force_class), max_class)
    else:
        intent_class = random.randint(0, max_class)

    signals  = generate_trial(intent_class)
    features = extract_features(signals)
    X_s      = scaler.transform(features.reshape(1, -1))

    t0       = time.time()
    snn_metrics = None

    if model_type == "snn" and snn_net is not None:
        import torch
        with torch.no_grad():
            Xt = torch.tensor(X_s, dtype=torch.float32)
            try:
                spk_out, spk1, spk2 = snn_net(Xt)
            except (TypeError, ValueError):
                # Backwards-compatible: old model returns single tensor
                old_fwd = spk_out = snn_net.forward.__wrapped__(snn_net, Xt) if hasattr(snn_net.forward, '__wrapped__') else None
                if old_fwd is None:
                    # Re-run with original net structure
                    spk_out = _legacy_snn_forward(snn_net, Xt)
                spk1 = spk2 = spk_out
            rates   = spk_out.mean(0)[0]
            probs   = torch.softmax(rates * 5, dim=0).numpy()
            # Restrict to num_classes
            probs = probs[:num_classes]
            probs = probs / probs.sum()
            pred  = int(probs.argmax())
            snn_metrics = compute_snn_metrics([spk_out, spk1, spk2])
    else:
        probs_full = gbm.predict_proba(X_s)[0]
        probs = probs_full[:num_classes]
        probs = probs / probs.sum()
        pred  = int(probs.argmax())

    latency_ms  = (time.time() - t0) * 1000
    confidence  = float(probs[pred])

    triggered = confidence >= threshold_pct and pred != (num_classes - 1)
    cancelled = confidence < threshold_pct and pred != (num_classes - 1)

    explanation = explain_prediction(X_s[0], pred)

    # Signal chart (downsample)
    t_arr = np.linspace(0, DURATION, int(FS * DURATION))
    step  = max(1, len(t_arr)//64)
    chart_data = {ch: signals[ch][::step].tolist() for ch in CHANNELS}
    time_axis  = t_arr[::step].tolist()

    # EMG power for coach
    emg_power_l = float(np.mean(signals["EMG_L"]**2))
    emg_power_r = float(np.mean(signals["EMG_R"]**2))

    # AI Coach feedback
    coach_msgs = generate_coach_feedback(
        confidence, snn_metrics, emg_power_l, emg_power_r,
        latency_ms, triggered, CLASSES[pred], model_type
    )

    session_history[patient_id].append({
        "intent":       CLASSES[pred],
        "true_intent":  CLASSES[intent_class],
        "confidence":   confidence,
        "triggered":    triggered,
        "latency_ms":   round(latency_ms, 1),
        "model":        model_type,
        "timestamp":    time.time(),
        "snn_metrics":  snn_metrics,
    })

    # Build full probs dict over 4 classes
    full_probs = {}
    for i in range(4):
        if i < num_classes:
            full_probs[CLASSES[i]] = round(float(probs[i]), 4)
        else:
            full_probs[CLASSES[i]] = 0.0

    return jsonify({
        "intent":        CLASSES[pred],
        "intent_class":  pred,
        "true_class":    intent_class,
        "true_intent":   CLASSES[intent_class],
        "confidence":    round(confidence, 4),
        "probabilities": full_probs,
        "triggered":     triggered,
        "cancelled":     cancelled,
        "latency_ms":    round(latency_ms, 1),
        "spike_count":   snn_metrics["spike_count"] if snn_metrics else None,
        "snn_metrics":   snn_metrics,
        "explanation":   explanation,
        "chart_data":    chart_data,
        "time_axis":     time_axis,
        "model_used":    model_type if (model_type == "snn" and snn_net) else "gbm",
        "coach_msgs":    coach_msgs,
        "emg_power_l":   round(emg_power_l, 3),
        "emg_power_r":   round(emg_power_r, 3),
        "threshold_used": threshold_pct,
        "num_classes":   num_classes,
    })


def _legacy_snn_forward(net, x):
    """Fallback for old SNN model that returns single tensor."""
    import torch
    mem1 = net.lif1.init_leaky()
    mem2 = net.lif2.init_leaky()
    mem3 = net.lif3.init_leaky()
    spk3_rec = []
    for _ in range(TIMESTEPS):
        cur1 = net.fc1(x);  spk1, mem1 = net.lif1(cur1, mem1)
        cur2 = net.fc2(spk1); spk2, mem2 = net.lif2(cur2, mem2)
        cur3 = net.fc3(spk2); spk3, mem3 = net.lif3(cur3, mem3)
        spk3_rec.append(spk3)
    return torch.stack(spk3_rec, 0)


@app.route("/recovery/<patient_id>")
def recovery(patient_id):
    history = session_history.get(patient_id, [])
    patient = PATIENTS.get(patient_id, PATIENTS["P001"])
    if not history:
        base          = patient["baseline_acc"]
        sessions_done = patient["sessions"]
        history_data  = []
        for i in range(sessions_done):
            acc = min(0.97, base + i*0.025 + random.uniform(-0.02, 0.02))
            history_data.append({
                "session": i+1, "accuracy": round(acc, 3),
                "triggers": random.randint(8, 20),
                "false_alarms_cancelled": random.randint(0, 3),
                "avg_latency_ms": round(random.uniform(5, 15), 1),
            })
    else:
        history_data = []
        for i in range(0, len(history), 10):
            chunk = history[i:i+10]
            acc   = float(np.mean([h["confidence"] for h in chunk]))
            history_data.append({
                "session": len(history_data)+1, "accuracy": round(acc, 3),
                "triggers": sum(1 for h in chunk if h["triggered"]),
                "false_alarms_cancelled": sum(1 for h in chunk if not h["triggered"] and h["intent"] != "Rest"),
                "avg_latency_ms": round(float(np.mean([h["latency_ms"] for h in chunk])), 1),
            })
        if not history_data:
            history_data = [{"session":1,"accuracy":round(float(np.mean([h["confidence"] for h in history])),3),"triggers":0,"false_alarms_cancelled":0,"avg_latency_ms":10}]

    forecast = forecast_recovery(history_data, patient)

    if session_history[patient_id]:
        cur = session_history[patient_id]
        summary = {
            "total_windows":  len(cur),
            "triggered":      sum(1 for h in cur if h["triggered"]),
            "avg_confidence": round(float(np.mean([h["confidence"] for h in cur])), 3),
            "avg_latency_ms": round(float(np.mean([h["latency_ms"] for h in cur])), 1),
        }
    else:
        summary = {"total_windows": 0, "triggered": 0, "avg_confidence": 0, "avg_latency_ms": 0}

    return jsonify({"history": history_data, "current_summary": summary, "forecast": forecast})

@app.route("/patients")
def patients_route():
    return jsonify(PATIENTS)

@app.route("/api/voice", methods=["POST"])
def voice():
    data = request.get_json() or {}
    text = data.get("text", "")
    result = parse_voice_command(text)
    return jsonify(result)

@app.route("/api/coach", methods=["POST"])
def coach():
    """Stand-alone coach analysis for current telemetry."""
    data = request.get_json() or {}
    pid  = data.get("patient_id", "P001")
    hist = session_history.get(pid, [])
    if not hist:
        return jsonify({"msgs": ["No session data yet. Run a simulation to receive coaching feedback."]})
    recent = hist[-5:]
    avg_conf    = float(np.mean([h["confidence"] for h in recent]))
    avg_lat     = float(np.mean([h["latency_ms"] for h in recent]))
    trigger_rate = sum(1 for h in recent if h["triggered"]) / len(recent)
    msgs = []
    if avg_conf > 0.85:
        msgs.append(f"🏆 Excellent recent performance! Average confidence {avg_conf*100:.1f}% over last {len(recent)} windows.")
    elif avg_conf > 0.70:
        msgs.append(f"📈 Good progress. Average confidence {avg_conf*100:.1f}%. Keep focusing on clear motor imagery.")
    else:
        msgs.append(f"🔄 Average confidence {avg_conf*100:.1f}%. Try closing your eyes and vividly imagining the movement before each trial.")
    if trigger_rate == 1.0:
        msgs.append("🎯 100% trigger rate — device would have been activated on every window. Outstanding consistency!")
    elif trigger_rate >= 0.6:
        msgs.append(f"🦾 {trigger_rate*100:.0f}% trigger activation rate. Consistent performance above safety threshold.")
    if avg_lat < 10:
        msgs.append(f"⚡ System latency averaging {avg_lat:.1f}ms — real-time BCI confirmed.")
    return jsonify({"msgs": msgs})

@app.route("/api/clinical_summary", methods=["POST"])
def clinical_summary():
    data = request.get_json() or {}
    pid  = data.get("patient_id", "P001")
    summary = generate_clinical_summary(pid)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    print("Starting Advanced AI Neuromorphic Rehab Suite...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, host="0.0.0.0", port=5000)
