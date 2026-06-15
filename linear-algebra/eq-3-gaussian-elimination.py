"""EQ-3: GAUSSIAN ELIMINATION - Combined Forward and Backward Elimination

This module combines EQ-1 (Row Echelon Form) and EQ-2 (Reduced Row Echelon Form) to perform 
full Gaussian elimination, transforming a system of linear equations into reduced row echelon form 
and directly providing the solution vector.

Process:
    1. Forward Elimination (EQ-1): Transform to Row Echelon Form (REF)
    2. Back Substitution (EQ-2): Transform to Reduced Row Echelon Form (RREF)
    3. Extract solution from the final RREF matrix
"""

import numpy as np
from eq_1_row_echelon_form import (
    augmented_matrix,
    get_index_first_non_zero_value_from_row,
    get_index_first_non_zero_value_from_column,
    swap_rows,
    row_echelon_form
)
from eq_2_reduced_row_echelon_form_or_back_substitution import back_substitution


def gaussian_elimination(A, B):
    """
    Perform complete Gaussian elimination (forward elimination + back substitution) to solve
    a system of linear equations Ax = B.

    Parameters:
    - A (numpy.array): Coefficient matrix (n x n)
    - B (numpy.array): Constants vector (n x 1)

    Returns:
    numpy.array: Solution vector x, where A @ x = B
    
    Process Flow:
    =============
    INPUT: System of linear equations represented as Ax = B
           Example: 2x + y - z = 8
                   -3x - y + 2z = -11
                   -2x + y + 2z = -3
    
    STEP 1 - Forward Elimination (EQ-1: ROW ECHELON FORM):
    -------
    Transforms augmented matrix [A|B] to upper triangular form with leading 1s
    [[1,   0.5, -0.5,  4  ],
     [0,   1,   -1.2, 3.5],
     [0,   0,    1,    2  ]]
    
    STEP 2 - Back Substitution (EQ-2: REDUCED ROW ECHELON FORM):
    --------
    Eliminates all non-zero elements above the pivots
    [[1,   0,    0,    2  ],
     [0,   1,    0,    3  ],
     [0,   0,    1,    5  ]]
    
    STEP 3 - Extract Solution:
    -------------------------
    Solution vector x = [2, 3, 5] from the last column of RREF matrix
    Represents: x = 2, y = 3, z = 5
    
    Real-World Applications:
    =======================
    - Circuit Analysis: Solve for currents in complex networks using Kirchhoff's laws
    - Structural Mechanics: Calculate forces and stresses in beams and structures
    - Economics: Solve supply-demand equilibrium models and market pricing
    - Machine Learning: Linear regression uses Gaussian elimination to find optimal parameters
    - Physics: Solve force equilibrium problems and system dynamics
    - Computer Graphics: Transform and project 3D coordinates
    """
    
    # Step 1: Forward Elimination - Convert to Row Echelon Form
    M_ref = row_echelon_form(A, B)
    
    if isinstance(M_ref, str) and M_ref == 'Singular system':
        return M_ref
    
    # Step 2: Back Substitution - Convert to Reduced Row Echelon Form and extract solution
    solution = back_substitution(M_ref)
    
    return solution


if __name__ == "__main__":
    # Example: System of 3 linear equations with 3 unknowns
    # 2x + y - z = 8
    # -3x - y + 2z = -11
    # -2x + y + 2z = -3
    
    A = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)
    
    B = np.array([[8], [-11], [-3]], dtype=float)
    
    print("="*80)
    print("EQ-3: GAUSSIAN ELIMINATION - Complete Solution Process")
    print("="*80)
    print()
    
    # Display original system
    print("ORIGINAL SYSTEM OF EQUATIONS:")
    print("-" * 80)
    print("2x + y - z = 8")
    print("-3x - y + 2z = -11")
    print("-2x + y + 2z = -3")
    print()
    
    print("AUGMENTED MATRIX [A|B]:")
    augmented = np.hstack([A, B])
    print(augmented)
    print()
    
    # ===========================
    # STEP 1: Forward Elimination
    # ===========================
    print("="*80)
    print("STEP 1: FORWARD ELIMINATION → ROW ECHELON FORM (EQ-1)")
    print("="*80)
    
    M_ref = row_echelon_form(A, B)
    print("\nAfter forward elimination (upper triangular form with pivots = 1):")
    print(M_ref)
    print()
    print("Properties of Row Echelon Form (REF):")
    print("  ✓ Upper triangular matrix")
    print("  ✓ All pivots (leading entries) equal 1")
    print("  ✓ All entries below each pivot are 0")
    print("  ✓ Ready for back substitution")
    print()
    
    # ===========================
    # STEP 2: Back Substitution
    # ===========================
    print("="*80)
    print("STEP 2: BACK SUBSTITUTION → REDUCED ROW ECHELON FORM (EQ-2)")
    print("="*80)
    
    M_rref = M_ref.copy()
    num_rows = M_rref.shape[0]
    
    # Perform back substitution to get RREF
    for row in reversed(range(num_rows)):
        index = get_index_first_non_zero_value_from_row(M_rref, row)
        for j in range(row):
            value = M_rref[j, index]
            M_rref[j] = M_rref[j] - value * M_rref[row]
    
    print("\nAfter back substitution (identity matrix with solution):")
    print(M_rref)
    print()
    print("Properties of Reduced Row Echelon Form (RREF):")
    print("  ✓ Identity matrix in coefficient part")
    print("  ✓ Solution directly readable from last column")
    print("  ✓ All entries above and below pivots are 0")
    print()
    
    # ===========================
    # STEP 3: Extract Solution
    # ===========================
    print("="*80)
    print("STEP 3: EXTRACT SOLUTION")
    print("="*80)
    print()
    
    solution = M_rref[:, -1]
    print(f"Solution Vector x = {solution}")
    print()
    print("Variable Values:")
    print(f"  x = {solution[0]:.6f}")
    print(f"  y = {solution[1]:.6f}")
    print(f"  z = {solution[2]:.6f}")
    print()
    
    # ===========================
    # Verification
    # ===========================
    print("="*80)
    print("VERIFICATION: Ax = B")
    print("="*80)
    
    result = A @ solution.reshape(-1, 1)
    print("\nA @ solution =")
    print(result)
    print("\nExpected B =")
    print(B)
    print()
    
    # Check if solution is correct
    if np.allclose(result, B):
        print("✓ SOLUTION VERIFIED: Ax = B is satisfied!")
    else:
        print("✗ Solution verification failed!")
    print()
    
    # ===========================
    # Real-World Application
    # ===========================
    print("="*80)
    print("REAL-WORLD APPLICATION")
    print("="*80)
    print()
    print("Gaussian elimination is fundamental in solving systems of linear equations found in:")
    print()
    print("1. CIRCUIT ANALYSIS")
    print("   - Solve for currents using Kirchhoff's voltage/current laws")
    print("   - Analyze complex multi-loop networks")
    print()
    print("2. STRUCTURAL MECHANICS")
    print("   - Calculate internal forces and stresses in structures")
    print("   - Solve equilibrium equations in static analysis")
    print()
    print("3. ECONOMICS & MARKET ANALYSIS")
    print("   - Solve supply-demand equilibrium systems")
    print("   - Calculate equilibrium prices and quantities")
    print()
    print("4. MACHINE LEARNING")
    print("   - Linear regression: Find optimal weights minimizing loss")
    print("   - Neural networks: Solve systems during backpropagation")
    print()
    print("5. PHYSICS")
    print("   - Force and acceleration calculations")
    print("   - System dynamics and motion equations")
    print()
    print("6. COMPUTER GRAPHICS")
    print("   - 3D transformations and coordinate projections")
    print("   - Rendering and visualization calculations")
    print()
