import pickle 
import sys
sys.stdout.reconfigure(line_buffering=True)

gate = sys.argv[1]
print(gate)

data=[]
if gate == "or":
    data = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    model_filename = "or_model.pkl"
elif gate == "and":
    data = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    model_filename = "and_model.pkl"
else:
    print("Invalid gate type. Use 'and' or 'or'.")
    sys.exit(1)

w1=0.3
w2=0.2
bias=-0.2
learning_rate=0.1

def step(x):
    return 1 if x>=0 else 0

def predict(x1,x2):
    z=w1*x1+w2*x2+bias
    return step(z)

for epoch in range(10):
    print(f"\nEpoch {epoch+1}", flush=True)
    error_count=0

    for x1,x2,target in data:
        z=w1*x1+w2*x2+bias
        prediction=step(z)
        error=target - prediction
        
    
        if error != 0:
            w1+=error*learning_rate*x1
            w2+=error*learning_rate*x2
            bias+=error*learning_rate
            error_count += 1
        print(f"Input: ({x1}, {x2}) -> Prediction: {prediction}, Target: {target}",flush=True)
        print(f"Updated w1: {w1:.2f}, w2: {w2:.2f}, bias: {bias:.2f}",flush=True)
    
    print("\n--- Testing Trained Model ---")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            result = predict(x1, x2)
            print(f"{x1} AND {x2} = {result}")

    if error_count == 0:
        print("Training complete — all outputs correct.")
        break

    if error_count == 0:
        print("Training complete")
model={
    "w1":w1,
    "w2":w2,
    "bias":bias
}

with open(f"{model_filename}","wb") as f:
    pickle.dump(model,f)

print(f"Model trained and saved as {gate}_model.pkl")
