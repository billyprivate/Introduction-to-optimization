import numpy as np

def simplex_method(C, A, b, epsilon=1e-6):
    m, n = A.shape
    assert len(C) == n and len(b) == m, "Input dimensions do not match."

    # Add slack variables
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:-1, :n] = A
    tableau[:-1, n:n+m] = np.eye(m)
    tableau[:-1, -1] = b
    tableau[-1, :n] = -C

    while True:
        # Check for optimality
        if all(tableau[-1, :-1] >= -epsilon):
            break

        # Choose entering variable (pivot column)
        pivot_col = np.argmin(tableau[-1, :n])

        # Check for unboundedness
        if all(tableau[:-1, pivot_col] <= epsilon):
            return "The method is not applicable!"

        # Choose departing variable (pivot row)
        pivot_row = np.argmin(tableau[:-1, -1] / tableau[:-1, pivot_col])

        # Perform pivot operation
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_val
        for i in range(m + 1):
            if i != pivot_row:
                multiplier = tableau[i, pivot_col]
                tableau[i, :] -= multiplier * tableau[pivot_row, :]

    solution = tableau[:-1, -1]
    objective_value = -tableau[-1, -1]

    return solution, objective_value

def get_user_input():
    try:
        n = int(input("Enter the number of decision variables (n): "))
        m = int(input("Enter the number of constraints (m): "))

        print("Enter the coefficients of the objective function (C):")
        C = np.array([float(input(f"C[{i+1}]: ")) for i in range(n)])

        print("Enter the coefficients of the constraint matrix (A) row by row:")
        A = np.array([[float(input(f"A[{i+1}][{j+1}]: ")) for j in range(n)] for i in range(m)])

        print("Enter the right-hand side values (b):")
        b = np.array([float(input(f"b[{i+1}]: ")) for i in range(m)])

        epsilon = float(input("Enter the approximation accuracy (epsilon): "))

        return C, A, b, epsilon
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return get_user_input()

if __name__ == "__main__":
    C, A, b, epsilon = get_user_input()

    result = simplex_method(C, A, b, epsilon)

    if isinstance(result, tuple):
        solution, objective_value = result
        n = len(C)  # Number of decision variables
        original_solution = solution[:n]  # Extrac
        print("Solution vector x*:", original_solution)
        print("Objective function value:", abs(objective_value))
    else:
        print(result)
3
