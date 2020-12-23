import numpy as np
# Defining batch of inputs with 2 features per input
input_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

target_output = np.array([[0, 1, 1, 1]])
target_output = target_output.reshape(4, 1)

# Defining weights:

weights = np.array([[0.1], [0.2]])
print(weights.shape)

bias = 0.3

lr = 0.05


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x)

for epoch in range(10000):
    inputs = input_features
    print("Input features before dot product \n", input_features)

    print("-----------------------------------")

    # Feedforward input:
    in_o = np.dot(inputs, weights) + bias

    print("Input features after dot product \n", in_o)
    # Feedforward output:
    out_o = sigmoid(in_o)
    print("-----------------------------------")
    print("Input features after sigmoid function ", out_o)
    # Backpropagation
    # Calculating error
    error = out_o - target_output

    # Going with the formula:
    x = error.sum()
    print(x)

    # Calculating derivative :
    derror_douto = error
    douto_dino = sigmoid_der(out_o)

    # Multiplying individual derivatives :
    deriv = derror_douto * douto_dino

    #Multiplying with the 3rd individual derivative:
    # Finding the transpose of input_features:
    inputs = input_features.T
    deriv_final = np.dot(inputs, deriv)

    # Updating the weights values :
    weights -= lr * deriv_final

    # Updating the bias the weight value :
    for i in deriv:
        bias -= lr * i


print("\nUpdated weights: \n")
print(weights)
print("Updated bias: \n")
print(bias)

# Taking inputs :
single_point = np.array([0, 0])

# 1st step:
result1 = np.dot(single_point, weights) + bias

# 2nd step:
result2 = sigmoid(result1)

# Print final result
print(result2)

