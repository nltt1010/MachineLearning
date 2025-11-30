from matrix import matrix
import math

class Layer:
    def __init__(self):
        pass
    def forward(self, input_data):
        raise NotImplementedError("Forward method not implemented.")
    def backward(self, output_error, learning_rate):
        raise NotImplementedError("Backward method not implemented.")
    
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, Z):
        self.Z = Z
        sigmoid_data = []
        for row in Z.data:
            new_row = [1.0 / (1.0 + math.exp(-z_val)) for z_val in row]
            sigmoid_data.append(new_row)
        sigmoid_matrix = matrix(sigmoid_data)
        self.output = sigmoid_matrix
        return sigmoid_matrix
    
    def backward(self, dA):
        sigmoid_prime_data = []
        for row in self.output:
            new_row = [val * (1.0 - val) for val in row]
            sigmoid_prime_data.append(new_row)
        sigmoid_prime = matrix(sigmoid_prime_data)
        # Chain Rule
        dZ_data = []
        for i in range(dA.shape()[0]):
            new_row = []
            for j in range(dA.shape()[1]):
                new_row.append(dA.data[i][j] * sigmoid_prime.data[i][j])
            dZ_data.append(new_row)
        return matrix(dZ_data)
        
        
class ReLU(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, Z):
        self.Z = Z
        relu_data = [[max(0.0, z_val) for z_val in row] for row in Z.data]
        relu_matrix = matrix(relu_data)
        self.output = relu_matrix
        return relu_matrix
    
    def backward(self, dA):
        relu_prime_data = [[1.0 if z_val > 0 else 0.0 for z_val in row] for row in self.Z.data]
        relu_prime = matrix(relu_prime_data)
        # Chain Rule
        dZ_data = []
        for i in range(dA.shape()[0]):
            new_row = []
            for j in range(dA.shape()[1]):
                new_row.append(dA.data[i][j] * relu_prime.data[i][j])
            dZ_data.append(new_row)
        return matrix(dZ_data)
    
    
class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.W = matrix.random(self.input_size, self.output_size)
        self.b = matrix.zeros(1, self.output_size)
        
        self.dW = matrix.zeros(self.input_size, self.output_size)
        self.db = matrix.zeros(1, self.output_size)
        
    def forward(self, X):
        self.X = X
        Z = (X @ self.W) + self.b
        return Z
    
    def backward(self, dZ):
        m = self.X.shape()[0]
        
        self.dW = (self.X.transpose() @ dZ) * (1 / m)
        self.db = matrix([[sum(dZ.data[i][j] for i in range(dZ.shape()[0])) / m 
                           for j in range(dZ.shape()[1])]])
        dX = dZ @ self.W.transpose()
        return dX
    
    def update(self, learning_rate):
        self.W = self.W - (self.dW * learning_rate)
        self.b = self.b - (self.db * learning_rate)
        
        
if __name__ == "__main__":
    print("Example:")
    dense_layer = Dense(input_size=3, output_size=2)
    X = matrix([[1, 2, 3], [4, 5, 6]])
    print("Input X:")
    print(X)
    Z = dense_layer.forward(X)
    print("Output Z after forward pass:")
    print(Z)
    dZ = matrix([[0.1, 0.2], [0.3, 0.4]])
    dX = dense_layer.backward(dZ)
    print("Gradient dX after backward pass:")
    print(dX)
    dense_layer.update(learning_rate=0.01)
    print("Updated weights W:")
    print(dense_layer.W)
    print("Updated biases b:")
    print(dense_layer.b)