from matrix import matrix

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None  

    def _step_function(self, z):
        return 1.0 if z >= 0 else -1.0

    def _prepare_data(self, X):
        m, n = X.shape() 
        bias_col_data = [[1.0] for _ in range(m)] 
        X_with_bias_data = [
            bias_col_data[i] + X.data[i] 
            for i in range(m)
        ]
        return matrix(X_with_bias_data)
        
    def fit(self, X, y):
        X = self._prepare_data(X)
        m, n = X.shape()
        self.weights = matrix.zeros(n, 1) 
        for _ in range(self.n_iters):
            for i in range(m):
                x_i_data = [X.data[i]] 
                x_i = matrix(x_i_data).transpose() 
                z = (self.weights.transpose() @ x_i).data[0][0] 
                y_hat = self._step_function(z)
                y_i = y.data[i][0]
                
                if y_i != y_hat:
                    delta = y_i - y_hat 
                    update_scalar = self.lr * delta 
                    update_vector = x_i * update_scalar
                    self.weights = self.weights + update_vector   
        return self

    def predict(self, X):
        X = self._prepare_data(X)
        z_vector = X @ self.weights
        predictions_data = [
            [self._step_function(z_vector.data[i][0])] 
            for i in range(z_vector.shape()[0])
        ]
        return matrix(predictions_data)
    

if __name__ == "__main__":
    print("Example:")
    x_train = matrix([[0], [1], [2], [3], [4], [5]])
    y_train = matrix([[-1], [-1], [-1], [1], [1], [1]])
    x_test = matrix([[1.5], [3.5]])
    print("Training data X:")
    print(x_train)
    print("Training data y:")
    print(y_train)
    model = Perceptron(learning_rate=0.01, n_iterations=10)
    print("Learning rate: 0.1, Number of iterations: 10")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("Predictions:")
    print(predictions)