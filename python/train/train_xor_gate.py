from ast import Pow
import numpy as np
import math

gate=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
X=[[x,y] for x,y,_ in gate]
Y=[(z) for _,_,z in gate]
print(f"{X}{Y}")

X = np.array(X)
Y = np.array(Y).reshape(-1, 1)


np.random.seed(0)

# Weight matrix: Input (2) → Hidden (3)
w_hidden = np.random.randn(2, 3)

# Bias for hidden layer (1 bias per hidden neuron)
b_hidden = np.zeros((1, 3))

# Weight matrix: Hidden (3) → Output (1)
w_output = np.random.randn(3, 1)

# Bias for output layer
b_output = np.zeros((1, 1))

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def sigmoid_derivative(prediction):
  return prediction * (1-prediction)

#forward pass
epochs = 1000
lr = 0.1
for epoch in range(epochs):
    z1=X.dot(w_hidden)+b_hidden
    a1=sigmoid(z1)
    z2=a1.dot(w_output)+b_output
    a2=sigmoid(z2)

  #Back propagation
    error=Y-a2
    d_output=error*sigmoid_derivative(a2)
    d_hidden= d_output.dot(w_output.T) * sigmoid_derivative(a1)

    w_output += a1.T.dot(d_output) * lr
    b_output += np.sum(d_output, axis=0, keepdims=True) * lr

    w_hidden += X.T.dot(d_hidden) * lr
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr
    if epoch % 100 == 0:
        loss = np.mean(np.square(error))
        print(f"Epochs: {epoch} - Loss: {loss}")
        print(f"\n Epoch: {epoch} - Loss: {loss:.6f}")
        print("────────────────────────────────────────────")

        i = 0  # You are printing for the first input (0, 0 → 0)
        print(f" Input Pattern: {X[i].tolist()} → Target: {Y[i][0]}")

        print("\n🔹 Forward Pass")
        print(f"z1 (input to hidden layer): {z1[i]}")
        print(f"a1 (hidden activations):    {a1[i]}")
        print(f"z2 (input to output):       {z2[i][0]:.6f}")
        print(f"a2 (prediction):            {a2[i][0]:.6f}")

        print("\n Error & Gradients")
        print(f"Error:                      {error[i][0]:.6f}")
        print(f"d_output (δ_output):        {d_output[i]}")
        print(f"d_hidden (δ_hidden):        {d_hidden[i]}")

        print("\n Gradients (Before Weight Update)")
        print("Gradient w_output:")
        print(a1[i][:, np.newaxis] @ d_output[i][:, np.newaxis].T)

        print(f"Gradient b_output:          {d_output[i]}")
        
        print("Gradient w_hidden:")
        print(X[i][:, np.newaxis] @ d_hidden[i][:, np.newaxis].T)

        print(f"Gradient b_hidden:          {d_hidden[i]}")
        print("────────────────────────────────────────────")
   
    
import pickle

model = {
  "w_hidden": w_hidden,
  "b_hidden": b_hidden,
  "w_output": w_output,
  "b_output": b_output
}

with open("xor_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to xor_model.pkl")