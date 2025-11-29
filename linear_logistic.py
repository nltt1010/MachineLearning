from matrix import matrix

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.theta = None
        
    def _sigmoid(self, Z):
        sigmoid_data = []
        for row in Z.data:
            new_row = [1.0 / (1.0 + math.exp(-z_val)) for z_val in row]
            sigmoid_data.append(new_row)
        return matrix(sigmoid_data)

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
        self.theta = matrix.zeros(n, 1) 
        
        for _ in range(self.n_iters):
            Z = X @ self.theta 
            y_hat = self._sigmoid(Z) 
            error = y_hat - y 
            X_T = X.transpose()
            gradient = (X_T @ error) * (1 / m) 
            self.theta = self.theta - (gradient * self.lr)
        return self

    def predict(self, X):
        X = self._prepare_data(X)
        Z = X @ self.theta
        probabilities = self._sigmoid(Z)
        predictions_data = [
            [1] if prob.data[i][0] >= 0.5 else [0]
            for i in range(probabilities.shape()[0])
        ]
        return matrix(predictions_data)
    
if __name__ == "__main__":
    import math
    print("Example:")
    x_train = matrix([[0], [1], [2], [3], [4], [5]])
    y_train = matrix([[0], [0], [0], [1], [1], [1]])
    x_test = matrix([[1.5], [3.5], [5.0]])
    print("Training data X:")
    print(x_train)
    print("Training data y:")
    print(y_train)
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    print("Learning rate: 0.1, Number of iterations: 1000")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("Predictions:")
    print(predictions)