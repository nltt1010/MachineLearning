from matrix import matrix

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.lr = learning_rate 
        self.n_iters = n_iterations
        self.theta = None    

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
        
        for i in range(self.n_iters):
            y_hat = X @ self.theta
            error = y_hat - y 
            X_T = X.transpose()
            gradient = (X_T @ error) * (1 / m) 
            self.theta = self.theta - (gradient * self.lr)
        return self
        
    def predict(self, X):
        X_new = self._prepare_data(X)
        return X_new @ self.theta
    
if __name__ == "__main__":
    print("Example:")
    x_train = matrix([[1], [2], [3], [4], [5]])
    y_train = matrix([[3], [5], [7], [9], [11]])
    x_test = matrix([[6], [7]])
    print("Training data X:")
    print(x_train)
    print("Training data y:")
    print(y_train)
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    print("Lerning rate: 0.1, Number of iterations: 1000")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("Predictions:")
    print(predictions)