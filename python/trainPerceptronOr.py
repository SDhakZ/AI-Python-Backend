data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1)
]

# Initial values
w1 = 0.3
w2 = 0.1
bias = -0.2   # bias = -θ
learning_rate = 0.1

# Step function
def step(x):
    return 1 if x >= 0 else 0

# Prediction function
def predict(x1, x2):
    z = w1 * x1 + w2 * x2 + bias
    return step(z)

# Training loop
for epoch in range(10):
    print(f"\nEpoch {epoch + 1}")
    error_count = 0

    for x1, x2, target in data:
        z = w1 * x1 + w2 * x2 + bias
        prediction = step(z)
        error = target - prediction

        # Update weights if there is an error
        if error != 0:
            w1 += learning_rate * error * x1
            w2 += learning_rate * error * x2
            bias += learning_rate * error
            error_count += 1

        print(f"Input: ({x1}, {x2}) → Prediction: {prediction}, Target: {target}")
        print(f"Updated w1: {w1:.2f}, w2: {w2:.2f}, bias: {bias:.2f}")

    print("\n--- Testing Trained Model ---")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            result = predict(x1, x2)
            print(f"{x1} OR {x2} = {result}")

    if error_count == 0:
        print("✅ Training complete — all outputs correct.")
        break
