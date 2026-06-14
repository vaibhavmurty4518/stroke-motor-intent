import os, sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Retraining GBM model fresh on this server...")

# Find dataset
BASE = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE, "data", "eeg_emg_dataset.csv")
model_dir = os.path.join(BASE, "model")

if not os.path.exists(data_path):
    print("Dataset not found, generating...")
    sys.path.insert(0, os.path.join(BASE, "data"))
    from generate_eeg_emg import generate_dataset
    generate_dataset()

df = pd.read_csv(data_path)
feature_cols = [c for c in df.columns if c not in ("label", "class_name")]
X = df[feature_cols].values.astype("float32")
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

gbm = GradientBoostingClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
)
gbm.fit(X_train_s, y_train)

acc = (gbm.predict(X_test_s) == y_test).mean()
print(f"GBM Accuracy: {acc:.4f}")

# Save fresh models
os.makedirs(model_dir, exist_ok=True)
joblib.dump(gbm,    os.path.join(model_dir, "gbm_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

# Update meta.json
meta_path = os.path.join(model_dir, "meta.json")
with open(meta_path) as f:
    meta = json.load(f)
meta["gbm_accuracy"] = float(acc)
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print("Model retrained and saved successfully!")
