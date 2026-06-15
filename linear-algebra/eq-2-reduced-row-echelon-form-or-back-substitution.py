import numpy as np
from eq_1_row_echelon_form import (
    augmented_matrix,
    get_index_first_non_zero_value_from_row,
    get_index_first_non_zero_value_from_column,
    swap_rows
)


def back_substitution(M):
    """
    Perform back substitution on an augmented matrix (with unique solution) in reduced row echelon form to find the solution to the linear system.

    Parameters:
    - M (numpy.array): The augmented matrix in row echelon form with unitary pivots (n x n+1).

    Returns:
    numpy.array: The solution vector of the linear system.
    
    Example Walkthrough:
    -------------------
    Input augmented matrix in row echelon form:
    [[1, 2, 0, 10],
     [0, 1, 0, 5],
     [0, 0, 1, 3]]
    
    Step 1 (row=2): Substitution row = [0, 0, 1, 3], index = 2
    - Process row 1: [0, 1, 0, 5] - 0 * [0, 0, 1, 3] = [0, 1, 0, 5] (no change, value=0)
    - Process row 0: [1, 2, 0, 10] - 0 * [0, 0, 1, 3] = [1, 2, 0, 10] (no change, value=0)
    
    Step 2 (row=1): Substitution row = [0, 1, 0, 5], index = 1
    - Process row 0: [1, 2, 0, 10] - 2 * [0, 1, 0, 5] = [1, 0, 0, 0]
    
    Final RREF matrix:
    [[1, 0, 0, 2],
     [0, 1, 0, 5],
     [0, 0, 1, 3]]
    
    Solution vector: [2, 5, 3] (represents x=2, y=5, z=3)
    """
    
    # Make a copy of the input matrix to avoid modifying the original
    M = M.copy()

    # Get the number of rows (and columns) in the matrix of coefficients
    num_rows = M.shape[0]

    ### START CODE HERE ####
    
    # Iterate from bottom to top
    for row in reversed(range(num_rows)): 
        substitution_row = M[row]

        # Get the index of the first non-zero element in the substitution row. Remember to pass the correct value to the argument augmented.
        index = get_index_first_non_zero_value_from_row(M, row)

        # Iterate over the rows above the substitution_row
        for j in range(row): 

            # Get the row to be reduced. The indexing here is similar as above, with the row variable replaced by the j variable.
            row_to_reduce = M[j]

            # Get the value of the element at the found index in the row to reduce
            value = row_to_reduce[index]
            
            # Perform the back substitution step using the formula row_to_reduce -> row_to_reduce - value * substitution_row
            row_to_reduce = row_to_reduce - value * substitution_row

            # Replace the updated row in the matrix, be careful with indexing!
            M[j, :] = row_to_reduce

    ### END CODE HERE ####

    # Extract the solution from the last column
    solution = M[:, -1]
    
    return solution


if __name__ == "__main__":
    # Example usage with a system of linear equations
    # 2x + y - z = 8
    # -3x - y + 2z = -11
    # -2x + y + 2z = -3
    
    from eq_1_row_echelon_form import row_echelon_form
    
    A = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)
    
    B = np.array([[8], [-11], [-3]], dtype=float)
    
    # Step 1: Convert to row echelon form
    M_echelon = row_echelon_form(A, B)
    print("Row Echelon Form:")
    print(M_echelon)
    print()
    
    # Step 2: Perform back substitution to get the solution
    solution = back_substitution(M_echelon)
    print("Solution vector:")
    print(solution)
    print()
    
    # Verify the solution
    print("Verification (A @ solution):")
    print(A @ solution.reshape(-1, 1))
    print()
    
    # Real-world application in one line
    print("REAL-WORLD APPLICATION: Back substitution transforms row echelon form to reduced row echelon form, solving linear systems used in engineering (circuit analysis, structural mechanics), economics (supply-demand models), machine learning (linear regression), and physics (force equilibrium) to find exact values of unknowns like electrical currents, market prices, or physical parameters.")
