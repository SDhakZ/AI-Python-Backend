# predict_bmi.py

import pickle
import sys
import numpy as np
import os
import subprocess

# --- Activation ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# === Ensure model is trained ===
if not os.path.exists("bmi_model_mlp.pkl"):
    print("Training model...", flush=True)
    subprocess.run(["python", "train_bmi_model.py"], check=True)

# === Load trained model ===
with open("bmi_model_mlp.pkl", "rb") as f:
    model = pickle.load(f)

w1 = model["w1"]
b1 = model["b1"]
w2 = model["w2"]
b2 = model["b2"]
input_scaler = model["input_scaler"]
target_scaler = model["target_scaler"]

# === CLI input ===
if len(sys.argv) != 4:
    print("Usage: python predict_bmi.py <height_m> <weight_kg> <gender>")
    sys.exit(1)

height = float(sys.argv[1])
weight = float(sys.argv[2])
gender_str = sys.argv[3].lower()

if gender_str not in ["male", "female"]:
    print("Gender must be 'male' or 'female'")
    sys.exit(1)

gender = 0 if gender_str == "male" else 1

# === Prepare input ===
input_scaled = input_scaler.transform([[gender, height, weight]])

# === Forward pass using sigmoid ===
z1 = np.dot(input_scaled, w1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, w2) + b2
output = z2  # Linear output (no sigmoid on final layer)

# === Rescale prediction ===
predicted_bmi = target_scaler.inverse_transform(output)[0][0]
print(f"✅ Predicted BMI: {predicted_bmi:.2f}")
