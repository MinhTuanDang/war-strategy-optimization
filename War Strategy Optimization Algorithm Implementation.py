import numpy as np

# -------------------------
# Define Helper Functions
# -------------------------

def fitness(x):
    """
    Example fitness function.
    For demonstration purposes, we'll assume a simple function: the sum of the vector elements.
    Modify this function to suit your optimization problem.
    """
    return np.sum(x)

def update_weight(w, new_fitness, old_fitness):
    """
    Example weight update function.
    Here, we simply increment the weight by a small constant if there is an improvement.
    Adjust this update rule as needed.
    """
    delta = 0.1
    return w + delta

# -------------------------
# War Strategy Optimization
# -------------------------

def war_strategy_optimization(N, R, Tmax, rho_threshold, initial_weight):
    """
    Parameters:
        N             : Number of soldiers (agents)
        R             : Dimensionality of the search space
        Tmax          : Maximum number of iterations
        rho_threshold : Threshold to decide exploration vs. exploitation
        initial_weight: Initial weight assigned to each soldier
    """
    # === Step 1: Initialization ===
    # Initialize positions randomly (here, positions are chosen uniformly from -10 to 10)
    X = [np.random.uniform(-10, 10, R) for _ in range(N)]
    # Compute fitness for each soldier
    F = [fitness(x) for x in X]
    # Initialize weight for each soldier
    W = [initial_weight for _ in range(N)]
    
    # === Step 2: Determine Initial Leadership ===
    # Sort indices in descending order (assuming higher fitness is better)
    sorted_indices = np.argsort(F)[::-1]
    King   = sorted_indices[0]
    Chief  = sorted_indices[1] if N > 1 else sorted_indices[0]
    Weakest = sorted_indices[-1]
    
    # === Step 3: Main Optimization Loop ===
    t = 0
    while t < Tmax:
        for i in range(N):
            # Generate random numbers for the decision process
            rho = np.random.rand()  # Uniform random number in [0, 1]
            r   = np.random.rand()  # Another random factor for scaling
            
            current_position = X[i]
            
            # Select a random soldier (different from i) for the exploration phase
            rand_index = np.random.randint(0, N)
            while rand_index == i:
                rand_index = np.random.randint(0, N)
            X_random = X[rand_index]
            
            # --- Movement Strategy ---
            if rho < rho_threshold:
                # Exploration Phase (global search)
                X_new = (current_position +
                         2 * rho * (X[King] - X_random) +
                         r * W[i] * (X[Chief] - current_position))
            else:
                # Exploitation Phase (local search)
                X_new = (current_position +
                         2 * rho * (X[Chief] - X[King]) +
                         r * W[i] * (X[King] - current_position))
            
            # --- Evaluate New Position ---
            F_new = fitness(X_new)
            
            # --- Accept or Reject the New Position ---
            # Here, we accept the new position if it offers an improvement.
            if F_new > F[i]:
                X[i] = X_new
                F[i] = F_new
                W[i] = update_weight(W[i], F_new, F[i])
        
        # --- Step 3.4: Update Leadership Positions ---
        sorted_indices = np.argsort(F)[::-1]
        King   = sorted_indices[0]
        Chief  = sorted_indices[1] if N > 1 else sorted_indices[0]
        
        # --- Step 3.5: Relocate the Weakest Soldier ---
        Weakest = sorted_indices[-1]
        X[Weakest] = np.random.uniform(-10, 10, R)
        F[Weakest] = fitness(X[Weakest])
        W[Weakest] = initial_weight
        
        t += 1

    # === Step 4: Output the Results ===
    print("Optimal Position (King):", X[King])
    print("Best Fitness:", F[King])
    return X[King], F[King]

# -------------------------
# Example Run
# -------------------------
if __name__ == "__main__":
    # Parameters
    N = 30              # Number of soldiers (agents)
    R = 5               # Dimension of the search space
    Tmax = 100          # Maximum iterations
    rho_threshold = 0.5 # Exploration vs. exploitation threshold
    initial_weight = 1.0

    war_strategy_optimization(N, R, Tmax, rho_threshold, initial_weight)