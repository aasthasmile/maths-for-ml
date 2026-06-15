import numpy as np


def augmented_matrix(A, B):
    """
    Create an augmented matrix by horizontally stacking two matrices A and B.

    Parameters:
    - A (numpy.array): First matrix.
    - B (numpy.array): Second matrix.

    Returns:
    - numpy.array: Augmented matrix obtained by horizontally stacking A and B.
    """
    augmented_M = np.hstack((A, B))
    return augmented_M


def get_index_first_non_zero_value_from_row(M, row, augmented=False):
    """
    Find the index of the first non-zero value in the specified row of the given matrix.

    Parameters:
    - M (numpy.array): The input matrix to search for non-zero values.
    - row (int): The index of the row to search.
    - augmented (bool): Pass this True if you are dealing with an augmented matrix, 
                        so it will ignore the constant values (the last column in the augmented matrix).

    Returns:
    int: The index of the first non-zero value in the specified row.
         Returns -1 if no non-zero value is found.
    """
    # Create a copy to avoid modifying the original matrix
    M = M.copy()

    # If it is an augmented matrix, then ignore the constant values
    if augmented == True:
        # Isolating the coefficient matrix (removing the constant terms)
        M = M[:, :-1]

    # Get the desired row
    row_array = M[row]
    for i, val in enumerate(row_array):
        # If finds a non zero value, returns the index. Otherwise returns -1.
        if not np.isclose(val, 0, atol=1e-5):
            return i
    return -1


def get_index_first_non_zero_value_from_column(M, column, starting_row):
    """
    Retrieve the index of the first non-zero value in a specified column of the given matrix.

    Parameters:
    - M (numpy.array): The input matrix to search for non-zero values.
    - column (int): The index of the column to search.
    - starting_row (int): The starting row index for the search.

    Returns:
    int: The index of the first non-zero value in the specified column, starting from the given row.
         Returns -1 if no non-zero value is found.
    """
    # Get the column array starting from the specified row
    column_array = M[starting_row:, column]
    for i, val in enumerate(column_array):
        # Iterate over every value in the column array. 
        # To check for non-zero values, you must always use np.isclose instead of doing "val == 0".
        if not np.isclose(val, 0, atol=1e-5):
            # If one non zero value is found, then adjust the index to match the correct index in the matrix and return it.
            index = i + starting_row
            return index
    # If no non-zero value is found below it, return -1.
    return -1


def swap_rows(M, row_index_1, row_index_2):
    """
    Swap rows in the given matrix.

    Parameters:
    - M (numpy.array): The input matrix to perform row swaps on.
    - row_index_1 (int): Index of the first row to be swapped.
    - row_index_2 (int): Index of the second row to be swapped.

    Returns:
    - numpy.array: Matrix with the specified rows swapped.
    """
    # Copy matrix M so the changes do not affect the original matrix.
    M = M.copy()
    # Swap indexes
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M


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
    
    # Create augmented matrix
    M = augmented_matrix(A, B.reshape(-1, 1))
    
    for row in range(num_rows):
        # Find the first non-zero value in the current row
        pivot_col = get_index_first_non_zero_value_from_row(M, row, augmented=True)
        
        if pivot_col == -1:
            # No non-zero value found, move to next row
            continue
        
        # Find the first non-zero value in the pivot column starting from the current row
        pivot_row = get_index_first_non_zero_value_from_column(M, pivot_col, row)
        
        if pivot_row == -1:
            # No non-zero value found in column, move to next row
            continue
        
        # Swap rows if necessary
        if pivot_row != row:
            M = swap_rows(M, row, pivot_row)
        
        # Scale the pivot row so the pivot element becomes 1
        pivot = M[row, pivot_col]
        M[row] = M[row] / pivot
        
        # Eliminate all values below the pivot
        for j in range(row + 1, num_rows):
            value_below_pivot = M[j, pivot_col]
            M[j] = M[j] - value_below_pivot * M[row]
    
    return M


if __name__ == "__main__":
    # Example usage
    A = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)
    
    B = np.array([[8], [-11], [-3]], dtype=float)
    
    result = row_echelon_form(A, B)
    print("Row Echelon Form:")
    print(result)
