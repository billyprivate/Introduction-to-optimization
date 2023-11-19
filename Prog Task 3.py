import numpy as np


def is_balanced(S, D):
    return sum(S) == sum(D)


def north_west_corner(S, D):
    i, j = 0, 0
    x = np.zeros_like(C)
    while i < len(S) and j < len(D):
        x[i, j] = min(S[i], D[j])
        S[i] -= x[i, j]
        D[j] -= x[i, j]
        if S[i] == 0:
            i += 1
        elif D[j] == 0:
            j += 1
    return x

# This function is used in vogels_approximation to find the minimum difference between the lowest and second-lowest costs.


def minimum_diff(costs, omit):
    """Calculate the minimum difference between the lowest and second-lowest costs."""
    lowest, second_lowest = np.inf, np.inf
    for i, c in enumerate(costs):
        if i in omit:
            continue
        elif c < lowest:
            second_lowest, lowest = lowest, c
        elif c < second_lowest:
            second_lowest = c
    return lowest if second_lowest == np.inf else second_lowest - lowest

# This function is used in vogels_approximation to find the column index of the lowest cost in a row.


def minimum_index_in_row(cost_table, i, supply_column, deleted_cols):
    """Return the column index of the lowest cost in a row."""
    costs = cost_table[i][:supply_column]
    costs_left = np.delete(costs, list(deleted_cols))
    lowest_cost = np.min(costs_left)
    j = list(set(np.where(costs == lowest_cost)[0]) - deleted_cols)[0]
    return j

# This function is used in vogels_approximation function to find the the row index of the lowest cost in a column.


def minimum_index_in_column(cost_table, j, demand_row, deleted_rows):
    """Return the row index of the lowest cost in a column."""
    costs = cost_table[:, j][:demand_row]
    costs_left = np.delete(costs, list(deleted_rows))
    lowest_cost = np.min(costs_left)
    i = list(set(np.where(costs == lowest_cost)[0]) - deleted_rows)[0]
    return i


def vogels_approximation(S, D, C):
    """Implement Vogel's approximation method for transportation problems."""
    num_rows, num_cols = C.shape
    cost_table = np.copy(C)
    supply_column, demand_row = num_cols, num_rows
    deleted_rows, deleted_cols = set(), set()
    assign_table = np.zeros_like(C)

    # Add diff column and row
    cost_table = np.append(cost_table, np.zeros((num_rows, 1)), axis=1)
    cost_table = np.append(cost_table, np.zeros((1, num_cols + 1)), axis=0)

    while len(deleted_rows) < num_rows and len(deleted_cols) < num_cols:
        # Update diff column and row
        for row in range(num_rows):
            if row not in deleted_rows:
                costs = cost_table[row, :num_cols]
                cost_table[row, num_cols] = minimum_diff(costs, deleted_cols)
        for col in range(num_cols):
            if col not in deleted_cols:
                costs = cost_table[:num_rows, col]
                cost_table[num_rows, col] = minimum_diff(costs, deleted_rows)

        # Choose the highest diff
        max_diff_row = max((r for r in range(num_rows) if r not in deleted_rows),
                           key=lambda x: cost_table[x, num_cols], default=-1)
        max_diff_col = max((c for c in range(num_cols) if c not in deleted_cols),
                           key=lambda x: cost_table[num_rows, x], default=-1)

        if max_diff_row == -1 or max_diff_col == -1:
            # Break the loop if there are no more valid rows or columns
            break

        if cost_table[max_diff_row, num_cols] >= cost_table[num_rows, max_diff_col]:
            i, j = max_diff_row, minimum_index_in_row(
                cost_table, max_diff_row, num_cols, deleted_cols)
        else:
            i, j = minimum_index_in_column(
                cost_table, max_diff_col, num_rows, deleted_rows), max_diff_col

        # Assign supply or demand
        supply_demand_min = min(S[i], D[j])
        assign_table[i, j] = supply_demand_min
        S[i] -= supply_demand_min
        D[j] -= supply_demand_min

        # Update deleted rows and columns
        if S[i] == 0 and i not in deleted_rows:
            deleted_rows.add(i)
        if D[j] == 0 and j not in deleted_cols:
            deleted_cols.add(j)

    return assign_table


def russells_approximation(S, D, C):
    num_rows, num_cols = C.shape
    x = np.zeros_like(C, dtype=float)
    u = np.full(num_rows, -np.inf)  # Max value in rows
    v = np.full(num_cols, -np.inf)  # Max value in columns
    remaining_supply = S.copy()
    remaining_demand = D.copy()

    while remaining_supply.sum() > 0 and remaining_demand.sum() > 0:
        # Update max values in rows and columns
        for i in range(num_rows):
            if remaining_supply[i] > 0:
                u[i] = max(C[i, :])
        for j in range(num_cols):
            if remaining_demand[j] > 0:
                v[j] = max(C[:, j])

        # Calculate Russell's cost difference and find max position
        max_value = -np.inf
        max_pos = (-1, -1)
        for i in range(num_rows):
            for j in range(num_cols):
                if remaining_supply[i] > 0 and remaining_demand[j] > 0:
                    russell_value = u[i] + v[j] - C[i, j]
                    if russell_value > max_value:
                        max_value = russell_value
                        max_pos = (i, j)

        # Allocate at max position
        i, j = max_pos
        allocation = min(remaining_supply[i], remaining_demand[j])
        x[i, j] = allocation
        remaining_supply[i] -= allocation
        remaining_demand[j] -= allocation

    return x


def string_to_array(input_string):
    return np.array([int(item) for item in input_string.split(',')])


# Getting input from user
supply_input = input("Enter supply vector (comma-separated): ")
demand_input = input("Enter demand vector (comma-separated): ")
num_rows = int(input("Enter the number of supply points: "))
num_cols = int(input("Enter the number of demand points: "))

S = string_to_array(supply_input)  # Convert to numpy array
D = string_to_array(demand_input)  # Convert to numpy array
C = np.zeros((num_rows, num_cols))

# Getting cost matrix input
print("Enter the cost matrix row by row (comma-separated):")
for i in range(num_rows):
    row_input = input(f"Row {i+1}: ")
    C[i] = string_to_array(row_input)

# Check if the problem is balanced
if not is_balanced(S, D):
    print("The problem is not balanced!")
else:
    print("Input Parameter Table:")
    print("Supply:", S)
    print("Demand:", D)
    print("Cost Matrix:\n", C)

    # Calculating solutions
    nw_solution = north_west_corner(S.copy(), D.copy())
    vogels_solution = vogels_approximation(S.copy(), D.copy(), C)
    russells_solution = russells_approximation(S.copy(), D.copy(), C)

    print("North-West Corner Solution:\n", nw_solution)
    print("Vogel's Approximation Solution:\n", vogels_solution)
    print("Russell's Approximation Solution:\n", russells_solution)
