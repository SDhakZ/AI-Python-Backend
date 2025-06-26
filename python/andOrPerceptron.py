def perceptron(x1, x2, w1, w2, bias):
    sum = w1 * x1 + w2 * x2 + bias
    return 1 if sum >= 0 else 0

# AND Gate
print("AND Gate:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        result = perceptron(x1, x2, 1, 1, -1.5)
        print(f"{x1} AND {x2} = {result}")

# OR Gate
print("\nOR Gate:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        result = perceptron(x1, x2, 1, 1, -0.5)
        print(f"{x1} OR {x2} = {result}")