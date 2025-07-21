import pickle
import sys
import os
import subprocess

# Expecting: python predict_gate.py <x1> <x2> <gate>
if len(sys.argv) != 4:
    print("Usage: python predict_gate.py <x1> <x2> <gate>")
    sys.exit(1)

x1 = int(sys.argv[1])
x2 = int(sys.argv[2])
gate = str(sys.argv[3])

print(x1, x2, gate)

def step(x):
    
    return 1 if x >= 0 else 0

model_path = f"{gate}_model.pkl"
train_script_path = "python/train/train_gate.py"

# Train if model not found
if not os.path.exists(model_path):
    print(f"{gate.upper()} model not found. Training model...", flush=True)
    try:
        subprocess.run(["python", train_script_path, gate], check=True)
    except subprocess.CalledProcessError as e:
        print("Error training model:", e)
        sys.exit(1)

# Load trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

w1 = model["w1"]
w2 = model["w2"]
bias = model["bias"]

print(w1,w2)
z = (x1 * w1) + (x2 * w2) + bias
output = step(z)

print(output)
