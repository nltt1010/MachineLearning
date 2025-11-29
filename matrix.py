class matrix:
    def __init__(self, data):
        if not data or not all(isinstance(row, list) for row in data):
            raise ValueError("Data must be a list of lists.")
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows must have the same length.")
        self.data = data
    
    def shape(self):
        return (len(self.data), len(self.data[0]) if self.data else 0)
    
    def __repr__(self):
        max_len = 0
        if self.data:
            max_len = max(len(str(val)) for row in self.data for val in row)
        repr_str = "[\n"
        for row in self.data:
            row_str = "  " + ", ".join(f"{val:{max_len}.4f}" if isinstance(val, (float)) else f"{val:{max_len}}" for val in row)
            repr_str += f"{row_str},\n"
        repr_str = repr_str.rstrip(',\n') + "\n]"
        return repr_str
    
    @staticmethod
    def zeros(rows, cols):
        return matrix([[0 for _ in range(cols)] for _ in range(rows)])

    @staticmethod
    def identity(size):
        data = matrix.zeros(size, size).data
        for i in range(size):
            data[i][i] = 1
        return matrix(data)

    def transpose(self):
        matrix_shape = self.shape()
        transposed_data = []
        for j in range(matrix_shape[1]):
            new_row = []
            for i in range(matrix_shape[0]):
                new_row.append(self.data[i][j])
            transposed_data.append(new_row)
        return matrix(transposed_data)

    
    def __add__(self, other):
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions to add.")
        result_data = [[self.data[i][j] + other.data[i][j] 
                        for j in range(self.shape()[1])] 
                       for i in range(self.shape()[0])]
        return matrix(result_data)
    
    def __sub__(self, other):
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions to subtract.")
        result_data = [[self.data[i][j] - other.data[i][j] 
                        for j in range(self.shape()[1])] 
                       for i in range(self.shape()[0])]
        return matrix(result_data)
    
    def __matmul__(self, other):
        if self.shape()[1] != other.shape()[0]:
            raise ValueError("Incompatible dimensions for matrix multiplication (@).")
        
        A_rows, A_cols = self.shape()
        B_rows, B_cols = other.shape()
        
        result_data = [[0 for _ in range(B_cols)] for _ in range(A_rows)]
        
        B_T_data = other.transpose().data

        for i in range(A_rows):
            for j in range(B_cols):
                sum_product = 0
                for k in range(A_cols): 
                    sum_product += self.data[i][k] * B_T_data[j][k]
                result_data[i][j] = sum_product
        return matrix(result_data)
    
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Multiplication (*) is only supported for scalar multiplication.")
            
        result_data = [[self.data[i][j] * scalar 
                        for j in range(self.shape()[1])] 
                       for i in range(self.shape()[0])]
        return matrix(result_data)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    
    def minor(self, row_idx, col_idx):
        new_data = []
        for i, row in enumerate(self.data):
            if i != row_idx:
                new_row = [val for j, val in enumerate(row) if j != col_idx]
                new_data.append(new_row)
        return matrix(new_data)
        
    def determinant(self):
        maxtrix_shape = self.shape()
        if maxtrix_shape[0] != maxtrix_shape[1]:
            raise ValueError("Determinant is only defined for square matrices.")
        n = maxtrix_shape[0]

        if n == 1:
            return self.data[0][0]
        elif n == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        else:
            det = 0
            for c in range(n):
                sub_matrix = self.minor(0, c) 
                minor_det = sub_matrix.determinant()
                cofactor = ((-1) ** c) * self.data[0][c] * minor_det
                det += cofactor
            return det
    
    def cofactor_matrix(self):
        maxtrix_shape = self.shape()
        n = maxtrix_shape[0]
        
        if n != maxtrix_shape[1]:
            raise ValueError("Cofactor matrix is only defined for square matrices.")
            
        cofactor_data = matrix.zeros(n, n).data
        
        for i in range(n):
            for j in range(n):
                sub_matrix = self.minor(i, j)
                cofactor_val = ((-1) ** (i + j)) * sub_matrix.determinant()
                cofactor_data[i][j] = cofactor_val
                
        return matrix(cofactor_data)
        
    def adjugate(self):
        cofactor_mat = self.cofactor_matrix()
        return cofactor_mat.transpose()
        
    def inverse(self):
        maxtrix_shape = self.shape()
        if maxtrix_shape[0] != maxtrix_shape[1]:
            raise ValueError("Inverse is only defined for square matrices.")
            
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        adj_matrix = self.adjugate()
        inv_matrix = adj_matrix * (1 / det)
        
        return inv_matrix
            
    
if __name__ == "__main__":
    print("-" * 50)
    m = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 
    n = matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    p = matrix([[4, 7], [2, 6]]) 
    print("Matrix n:")
    print(n)
    print(f"Shape of n: {n.shape()}")
    print(f"Transpose of n:\n{n.transpose()}")
    print("-" * 50)
    
    print(f"n @ n (Matrix Multiplication):\n{n @ n}")
    print(f"n + n:\n{n + n}")
    print(f"n - n:\n{n - n}")
    print(f"n * 10 (Scalar Multiplication):\n{n * 10}")
    print("-" * 50)

    print(f"Determinant of m ([1,2,3],[4,5,6],[7,8,9]): {m.determinant()}") 
    print(f"Determinant of n: {n.determinant()}")                          
    print(f"Determinant of p ([4,7],[2,6]): {p.determinant()}")           
    print("-" * 50)
    
    try:
        m_inv = m.inverse()
    except ValueError as e:
        print(f"Inverse of m failed: {e}") 
        
    p_inv = p.inverse()
    print(f"Inverse of p (2x2):\n{p_inv}")
    print(f"p @ p_inv:\n{p @ p_inv}") 
    print("-" * 50)

    n_inv = n.inverse()
    print(f"Inverse of n (3x3):\n{n_inv}")
    print(f"n @ n_inv:\n{n @ n_inv}")
    print("-" * 50)