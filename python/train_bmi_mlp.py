# train_bmi_model.py

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- Activation Functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# --- Load and preprocess dataset ---
df = pd.read_csv("bmi.csv")

df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df["height_m"] = df["Height"] / 100
df["bmi"] = df["Weight"] / (df["height_m"] ** 2)

# Features and target
X = df[["Gender", "height_m", "Weight"]].values
y = df[["bmi"]].values

# --- Scaling ---
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = input_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# --- Network configuration ---
input_size = 3
hidden_size = 4
output_size = 1
lr = 0.01
epochs = 1000

# --- Initialize weights and biases (Xavier for sigmoid) ---
limit1 = np.sqrt(6 / (input_size + hidden_size))
limit2 = np.sqrt(6 / (hidden_size + output_size))

w1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
b1 = np.zeros((1, hidden_size))
w2 = np.random.uniform(-limit2, limit2, (hidden_size, output_size))
b2 = np.zeros((1, output_size))

# --- Training loop ---
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X_train, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = z2  # Linear output for regression

    if np.isnan(a2).any() or np.isinf(a2).any():
        print(f"❌ NaN or Inf detected in epoch {epoch}")
        break

    # Backpropagation
    error = a2 - y_train
    d_z2 = error
    d_w2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, w2.T)
    d_z1 = d_a1 * sigmoid_derivative(z1)
    d_w1 = np.dot(X_train.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # Gradient clipping
    for grad in [d_w1, d_w2, d_b1, d_b2]:
        np.clip(grad, -1.0, 1.0, out=grad)

    # Update weights
    w1 -= lr * d_w1
    b1 -= lr * d_b1
    w2 -= lr * d_w2
    b2 -= lr * d_b2

    # Optional: print loss
    if epoch % 100 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# --- Save model ---
model = {
    "w1": w1,
    "b1": b1,
    "w2": w2,
    "b2": b2,
    "input_scaler": input_scaler,
    "target_scaler": target_scaler
}

with open("bmi_model_mlp.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ BMI regression model trained with sigmoid and saved as bmi_model_mlp.pkl")
