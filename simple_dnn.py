from matrix import matrix
import math

class SimpleDNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, n_iterations=100):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = matrix.random(self.input_size, self.hidden_size)
        self.W2 = matrix.random(self.hidden_size, self.output_size)
        
    def _sigmoid(self, Z):
        sigmoid_data = []
        for row in Z.data:
            new_row = [1.0 / (1.0 + math.exp(-z_val)) for z_val in row]
            sigmoid_data.append(new_row)
        return matrix(sigmoid_data)

    def _sigmoid_derivative(self, A):
        one_minus_A_data = [[1.0 - val for val in row] for row in A.data]
        one_minus_A = matrix(one_minus_A_data)
        result_data = [[A.data[i][j] * one_minus_A.data[i][j] 
                        for j in range(A.shape()[1])] 
                       for i in range(A.shape()[0])]
        return matrix(result_data)

    def forward(self, X):
        self.Z1 = X @ self.W1  
        self.A1 = self._sigmoid(self.Z1) 
        
        self.Z2 = self.A1 @ self.W2 
        self.A2 = self._sigmoid(self.Z2)
        return self.A2 
    
    def backward(self, X, y, y_hat):
        m = X.shape()[0]
        dZ2 = y_hat - y 
    
        dW2 = (self.A1.transpose() @ dZ2) * (1 / m)
        
    
        dA1 = dZ2 @ self.W2.transpose()
        sigmoid_prime_A1 = self._sigmoid_derivative(self.A1)
        dZ1_data = [[dA1.data[i][j] * sigmoid_prime_A1.data[i][j]
                     for j in range(dA1.shape()[1])]
                    for i in range(dA1.shape()[0])]
        dZ1 = matrix(dZ1_data) 
    
        dW1 = (X.transpose() @ dZ1) * (1 / m)
        
        self.W2 = self.W2 - (dW2 * self.lr)
        self.W1 = self.W1 - (dW1 * self.lr)
        
    def fit(self, X, y):
        for i in range(self.n_iters):
            y_hat = self.forward(X)
            self.backward(X, y, y_hat)
        return self
    
    def predict(self, X):
        return self.forward(X)
    
if __name__ == "__main__":
    print("Example:")
    x_train = matrix([[0,0], [0,1], [1,0], [1,1]])
    y_train = matrix([[0], [1], [1], [0]])  
    x_test = matrix([[0,0], [0,1], [1,0], [1,1]])
    print("Training data X:")
    print(x_train)
    print("Training data y:")
    print(y_train)
    model = SimpleDNN(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5, n_iterations=10000)
    print("Learning rate: 0.5, Number of iterations: 10000")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("Predictions:")
    print(predictions)
    
    