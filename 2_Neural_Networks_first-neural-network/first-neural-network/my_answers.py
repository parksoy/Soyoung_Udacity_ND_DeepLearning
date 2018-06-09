import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):# Set number of nodes in input, hidden and output layers, Initialize weights
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,(self.input_nodes, self.hidden_nodes)) #(3,2)
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,(self.hidden_nodes, self.output_nodes)) #(2,1)
        self.lr = learning_rate
        self.activation_function = lambda x : 1/(1+np.exp(-x)) #### TODO: Set self.activation_function to your implemented sigmoid function ####

    def train(self, features, targets): #''' Train the network on batch of features and targets.-features: 2D array, each row is one data record, each column is a feature-targets: 1D array of target values'''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X): #X= (3,) ''' Implement forward pass here-X: features batch'''
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) #(1,3)(3,2)=(1,2) signals into hidden layer# TODO: Hidden layer - Replace these values with your calculations.
        #print("X",X, X.shape)
        #print("self.weights_input_to_hidden",self.weights_input_to_hidden)
        #print("hidden_inputs", hidden_inputs)
        hidden_outputs = self.activation_function(hidden_inputs) # (1,2) signals from hidden layer:##Reviewer#1:You should not apply an activation in the final layer, because this is a regression problem.
        #print("hidden_outputs", hidden_outputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) #(1,2), (2,1)=(1,1) signals into final output layer # TODO: Output layer - Replace these values with your calculations.
        final_outputs = final_inputs # (1,1) signals from final output layer
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o): #''' Implement backpropagation-final_outputs: output from forward pass-y: target (i.e. label) batch-delta_weights_i_h: change in weights from input to hidden layers-delta_weights_h_o: change in weights from hidden to output layers'''
        error = y - final_outputs # (1,1) Output layer error is the difference between desired target and actual output.# TODO: Output error - Replace this value with your calculations.
        output_error_term = error  # (1,1) # TODO: Calculate the hidden layer's contribution to the error ##Reviewer#1 pointed

        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)  # (1,1), (1,2)=(1,2)
        #Reviewer 1 hinted * and np.dot operation are totally different!
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)  #(1,2)*(1,2)*()=(1,2) #TODO: Backpropagated error terms - Replace these values with your calculations.
        #[[-0.007632  0.008496]]= [[ 0.12 -0.04]] *[[-0.06 -0.18]]*(1-[[-0.06 -0.18]])

        delta_weights_i_h += hidden_error_term * X[:,None] # (1,2), (1,3)=(3,2) # Weight step (input to hidden)
        #[[-0.007632  0.008496]] * [[ 0.5]  = [[-0.003816   0.004248 ]
                                 #   [-0.2]    #[ 0.0015264 -0.0016992]
                                 #   [ 0.1]]   #[-0.0007632  0.0008496]]

        delta_weights_h_o += output_error_term * hidden_outputs[:,None]  # (1,1)*(1,2)=(1,2) # Weight step (hidden to output)
        #[[0.4]]* [[-0.06 -0.18]].T = (2,1)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records): #''' Update weights on gradient descent step-delta_weights_i_h: change in weights from input to hidden layers-delta_weights_h_o: change in weights from hidden to output layers-n_records: number of records'''
        self.weights_hidden_to_output += self.lr  * delta_weights_h_o / (n_records) # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr  * delta_weights_i_h / (n_records) # update input-to-hidden weights with gradient descent step

    def run(self, features): #features (1, 3) ''' Run a forward pass through the network with input features-features: 1D array of feature values'''
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) #(1,3),(3,2)=(1,2) )# signals into hidden layer# TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_outputs = self.activation_function(hidden_inputs) # (1,2)# signals from hidden layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) #(1,2)* (2,1)=(1,1)# signals into final output layer# TODO: Output layer - Replace these values with the appropriate calculations.
        final_outputs = final_inputs# signals from final output layer
        return final_outputs

#########################################################
# Set your hyperparameters here
##########################################################
iterations = 4000
learning_rate = 0.5
hidden_nodes = 28
output_nodes = 1
