import numpy as np
from numpy.linalg import norm, LinAlgError


def simplex_method(C, A, b, epsilon):
    m, n = A.shape
    assert len(C) == n and len(b) == m, "Input dimensions do not match."

    # Add slack variables and form the initial tableau
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
        pivot_col = np.argmin(tableau[-1, :-1])

        # Check for unboundedness
        if all(tableau[:-1, pivot_col] <= epsilon):
            return "The problem is unbounded!"

        # Choose departing variable (pivot row)
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        pivot_row = np.argmin(ratios)

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


def interior_point_method(C, A, b, epsilon, alpha, x):
    m, n = A.shape

    i = 1
    try:
        while True:
            v = x
            D = np.diag(x)
            AA = np.dot(A, D)
            cc = np.dot(D, C)
            I = np.eye(n)
            F = np.dot(AA, np.transpose(AA))
            FI = np.linalg.inv(F)

            if np.linalg.matrix_rank(AA) < m:
                return "The method is not applicable!"

            H = np.dot(np.transpose(AA), FI)
            P = np.subtract(I, np.dot(H, AA))
            cp = np.dot(P, cc)
            nu = np.absolute(np.min(cp))

            # Check if the problem does not have a solution
            if nu == 0:
                return "The problem does not have a solution."

            y = np.add(np.ones(n, float), (alpha/nu)*cp)
            yy = np.dot(D, y)
            x = yy

            print("In iteration", i, "we have x =", x, "\n")

            i += 1

            # Check for convergence based on the relative change in the solution
            if norm(np.subtract(yy, v), ord=2) < epsilon:
                break

    except LinAlgError as e:
        return f"Error: {str(e)}"

    # Calculate the maximum value of the objective function
    max_value = np.dot(C, x)

    return x, max_value

try:
    x = np.array(input("Enter coefficients of X (space separated): ").split(), dtype=float)
    C = np.array(input("Enter coefficients of the objective function C (space-separated): ").split(), dtype=float)
    A = np.array([input("Enter row of coefficients for constraint matrix A (space-separated): ").split() for _ in
                  range(int(input("Enter the number of constraints (rows of A): ")))], dtype=float)
    b = np.array(input("Enter the right-hand side vector b (space-separated) : ").split(), dtype=float)
    epsilon = float(input("Enter the approximation accuracy (epsilon): "))
    alpha = 0.5

    # Interior-Point method with alpha = 0.5
    result_0_5 = interior_point_method(C, A, b, epsilon, alpha, x)
    print("\nInterior-Point method with alpha = 0.5:")
    for i in range(len(list(result_0_5[0]))):
        print(f"x{i + 1}: {list(result_0_5[0])[i]}")
    print(f"Value of the function: {result_0_5[1]}")


    # Interior-Point method with alpha = 0.9
    alpha = 0.9
    result_0_9 = interior_point_method(C, A, b, epsilon, alpha, x)
    print("\nInterior-Point method with alpha = 0.9:")
    for i in range(len(list(result_0_9[0]))):
        print(f"x{i + 1}: {list(result_0_9[0])[i]}")
    print(f"Value of the function: {result_0_9[1]}")

    print("Simplex Method: ")
    result = simplex_method(C, A, b, epsilon)
    if isinstance(result, tuple):
        solution, objective_value = result
        n = len(C)  # Number of decision variables
        original_solution = solution[:n]  # Extrac
        print("Objective function value:", abs(objective_value))
    else:
        print(result)

except Exception as e:
    print(f"An error occurred: {str(e)}")
