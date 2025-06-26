# predict_bmi.py
import pickle
import sys
import numpy as np
import os
import subprocess

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
train_script_path = "python/train_model.py"
if not os.path.exists("bmi_model.pkl"):
    print(f"bmi model not found. Training model...", flush=True)
    try:
        subprocess.run(["python", "python/train_model.py"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error training model:", e)
        sys.exit(1)

# Load trained model
with open("bmi_model.pkl", "rb") as f:
    model = pickle.load(f)

w1 = model["w1"]
w2 = model["w2"]
bias = model["bias"]
input_scaler = model["input_scaler"]
target_scaler = model["target_scaler"]




# Get CLI input
if len(sys.argv) != 3:
    print("Usage: python predict_bmi.py <height_m> <weight_kg>")
    sys.exit(1)

height = float(sys.argv[1])
weight = float(sys.argv[2])

input_scaled = input_scaler.transform([[height, weight]])
x1, x2 = input_scaled[0] 

z = (x1 * w1) + (x2 * w2) + bias
output = sigmoid(z)

predicted_bmi = target_scaler.inverse_transform([[output]])[0][0]
print(f"{predicted_bmi:.2f}")
