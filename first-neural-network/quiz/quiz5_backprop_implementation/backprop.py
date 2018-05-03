import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5, size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        hidden_input = np.dot(x, weights_input_hidden)# TODO: Calculate the output
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_output,weights_hidden_output))

        ## Backward pass ##
        error = y - output# TODO: Calculate the network's prediction error
        output_error_term = error * output * (1 - output)# TODO: Calculate error term for the output unit

        ## propagate errors to hidden layer
        hidden_error = np.dot(output_error_term, weights_hidden_output)# TODO: Calculate the hidden layer's contribution to the error
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output) # TODO: Calculate the error term for the hidden layer

        del_w_hidden_output += output_error_term * hidden_output # TODO: Update the change in weights
        del_w_input_hidden += hidden_error_term * x[:, None]

    weights_input_hidden += learnrate * del_w_input_hidden / n_records # TODO: Update weights
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("e=", e)
            print("  out:", out)
            print("  Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("e=", e)
            print("  out:", out)
            print("  Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
print("\n\nAfter Traininng, \nout:\n",out)
print("out > 0.5", out > 0.5)
predictions = out > 0.5 #this converts to 1,0 target
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
