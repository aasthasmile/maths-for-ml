import numpy as np
from row_echelon_form import (
    augmented_matrix,
    get_index_first_non_zero_value_from_row,
    get_index_first_non_zero_value_from_column,
    swap_rows
)


def back_substitution(M):
    """
    Perform back substitution on an augmented matrix (with unique solution) in reduced row echelon form to find the solution to the linear system.
    
    This function reduces all elements above the pivot positions (diagonal elements) to zero, transforming the matrix into reduced row echelon form (RREF),
    which directly gives the solution to the linear system without further computation.

    Parameters:
    - M (numpy.array): The augmented matrix in row echelon form with unitary pivots (n x n+1).

    Returns:
    numpy.array: The solution vector of the linear system.
    """
    
    # Make a copy of the input matrix to avoid modifying the original
    M = M.copy()

    # Get the number of rows (and columns) in the matrix of coefficients
    num_rows = M.shape[0]

    ### START CODE HERE ####
    
    # Iterate from bottom to top
    # Example: If we have 3 rows, iterate in order: row 2, row 1, row 0
    for row in reversed(range(num_rows)): 
        substitution_row = M[row]
        # Example: substitution_row = [0, 1, 0, 5] (pivot at index 1, with solution 5)

        # Get the index of the first non-zero element in the substitution row
        index = get_index_first_non_zero_value_from_row(M, row)
        # Example: index = 1 (position of the leading 1 in the pivot)

        # Iterate over the rows above the substitution_row
        for j in range(row): 
            # Example: If row = 2, iterate j = 0, 1 (rows above row 2)

            # Get the row to be reduced
            row_to_reduce = M[j]
            # Example: row_to_reduce = [1, 2, 0, 10] (has a 2 at the pivot column)

            # Get the value of the element at the found index in the row to reduce
            value = row_to_reduce[index]
            # Example: value = 2 (the element to eliminate above the pivot)
            
            # Perform the back substitution step using the formula row_to_reduce -> row_to_reduce - value * substitution_row
            row_to_reduce = row_to_reduce - value * substitution_row
            # Example: [1, 2, 0, 10] - 2 * [0, 1, 0, 5] = [1, 0, 0, 0]
            # Result: The element above the pivot becomes 0

            # Replace the updated row in the matrix, be careful with indexing!
            M[j, :] = row_to_reduce
            # Example: After this, M[j] = [1, 0, 0, 0]

    ### END CODE HERE ####

    # Extract the solution from the last column
    # Example: If M = [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4]], solution = [2, 3, 4]
    solution = M[:, -1]
    
    return solution


if __name__ == "__main__":
    # Example usage with a system of linear equations
    # 2x + y - z = 8
    # -3x - y + 2z = -11
    # -2x + y + 2z = -3
    
    from row_echelon_form import row_echelon_form
    
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
    
    # Real-world application explanation
    print("="*70)
    print("REAL-WORLD APPLICATION:")
    print("="*70)
    print("Back substitution converts row echelon form to reduced row echelon form,")
    print("directly solving systems of equations found in engineering (circuit analysis,")
    print("structural mechanics), economics (supply-demand models), machine learning")
    print("(linear regression), and physics (force equilibrium) - providing exact")
    print("solutions where variables represent unknowns like currents, prices, or parameters.")
    print("="*70)
