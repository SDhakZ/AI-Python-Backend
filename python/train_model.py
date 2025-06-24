# train_model.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Load and preprocess data
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(root_path, "bmi.csv")
df = pd.read_csv(csv_path)
df.columns = ["gender", "height", "weight_kg", "bmi"]
df.drop(["gender", "bmi"], axis=1, inplace=True)
df["height_m"] = df["height"] / 100
df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)

features = df[['height_m', 'weight_kg']]
target = df[['bmi']]

input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
scaled_features = input_scaler.fit_transform(features)
scaled_target = target_scaler.fit_transform(target)

X = np.array(scaled_features)
y = np.array(scaled_target).flatten()

# Initialize weights
w1 = np.random.rand()
w2 = np.random.rand()
bias = np.random.rand()
learning_rate = 0.2
epochs = 1000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# Train perceptron
for epoch in range(epochs):
    for i in range(len(X)):
        x1, x2 = X[i]
        target_val = y[i]

        z = (x1 * w1) + (x2 * w2) + bias
        output = sigmoid(z)
        error = target_val - output

        d_output = error * sigmoid_derivative(output)
        w1 += learning_rate * d_output * x1
        w2 += learning_rate * d_output * x2
        bias += learning_rate * d_output

# Save model and scalers
model = {
    "w1": w1,
    "w2": w2,
    "bias": bias,
    "input_scaler": input_scaler,
    "target_scaler": target_scaler
}

with open("bmi_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as bmi_model.pkl")
