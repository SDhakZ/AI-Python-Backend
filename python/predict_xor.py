import sys
import numpy as np
import pickle

# Usage check
if len(sys.argv) != 3:
    print("Usage: python predict_gate.py <x1> <x2>")
    sys.exit(1)

# Parse inputs
x1 = int(sys.argv[1])
x2 = int(sys.argv[2])
x_input = np.array([[x1, x2]])

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step(x):
    return 1 if x >= 0.5 else 0  # threshold at 0.5 for sigmoid output


# Load trained model
with open("xor_model.pkl", "rb") as f:
    model = pickle.load(f)

w_hidden = model["w_hidden"]
b_hidden = model["b_hidden"]
w_output = model["w_output"]
b_output = model["b_output"]

# Forward pass
z1 = x_input.dot(w_hidden) + b_hidden
a1 = sigmoid(z1)
z2 = a1.dot(w_output) + b_output
a2 = sigmoid(z2)

# Output result
prediction = step(a2[0][0])
print(prediction)
