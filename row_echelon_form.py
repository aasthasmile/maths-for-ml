import numpy as np


def get_index_first_non_zero_value_from_column(M, row, col):
    """
    Find the index of the first non-zero value in a column below the current row.
    
    Args:
        M: The augmented matrix
        row: Current row index
        col: Column index
    
    Returns:
        Index of the first non-zero value below the current row
    """
    for i in range(row, len(M)):
        if not np.isclose(M[i, col], 0):
            return i
    return row


def row_echelon_form(A, B):
    """
    Convert a system of linear equations to row echelon form.
    
    Args:
        A: Coefficient matrix (n x n)
        B: Constants vector (n x 1)
    
    Returns:
        Augmented matrix in row echelon form, or 'Singular system' if matrix is singular
    """
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):
        return 'Singular system'
    
    A = A.copy().astype('float64')
    B = B.copy().astype('float64')
    num_rows = len(A)
    M = np.hstack([A, B.reshape(-1, 1)])
    
    for row in range(num_rows):
        pivot_candidate = M[row, row]
        
        if np.isclose(pivot_candidate, 0):
            first_non_zero = get_index_first_non_zero_value_from_column(M, row, row)
            M[[row, first_non_zero]] = M[[first_non_zero, row]]
            pivot = M[row, row]
        else:
            pivot = pivot_candidate
        
        M[row] = (1 / pivot) * M[row]
        
        for j in range(row + 1, num_rows):
            value_below_pivot = M[j, row]
            M[j] = M[j] - value_below_pivot * M[row]
    
    return M


if __name__ == "__main__":
    # Example usage
    A = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)
    
    B = np.array([8, -11, -3], dtype=float)
    
    result = row_echelon_form(A, B)
    print("Row Echelon Form:")
    print(result)
