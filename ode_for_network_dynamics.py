import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import comb
# Import added for Gaussian smoothing
from scipy.ndimage import gaussian_filter

# Define states (for clarity)
T = 0  # Truthful
H = 1  # Hallucinating
D = 2  # Does not know
STATES = [T, H, D]
NUM_STATES = len(STATES)

# --- Compute and Smooth Q from Adjacency Matrix ---
def compute_and_smooth_q(adjacency_matrix, sigma=1.0):
    """
    Computes the empirical joint degree distribution Q(l, m) from an
    adjacency matrix and applies Gaussian smoothing.

    Args:
        adjacency_matrix (np.ndarray): A square matrix where A[i, j] = 1
                                       indicates an edge from j to i.
        sigma (float): Standard deviation for the Gaussian kernel used in smoothing.

    Returns:
        tuple: (Q_smooth, L_max, M_max)
            Q_smooth (np.ndarray): The smoothed, normalized joint degree distribution.
            L_max (int): The maximum observed in-degree.
            M_max (int): The maximum observed out-degree.
    """
    A = np.asarray(adjacency_matrix)
    N = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    # Calculate in-degrees (l) and out-degrees (m)
    # l_i = sum over columns for row i (incoming edges)
    in_degrees = np.sum(A, axis=1)
    # m_i = sum over rows for column i (outgoing edges)
    out_degrees = np.sum(A, axis=0)

    # Find max degrees
    L_max = np.max(in_degrees)
    M_max = np.max(out_degrees)

    # Compute empirical counts
    Q_counts = np.zeros((L_max + 1, M_max + 1))
    for i in range(N):
        l = in_degrees[i]
        m = out_degrees[i]
        Q_counts[l, m] += 1

    # Normalize to get empirical distribution
    if N > 0:
        Q_empirical = Q_counts / N
    else:
        Q_empirical = Q_counts # Should remain all zeros

    print(f"Empirical L_max = {L_max}, M_max = {M_max}")
    # print("Empirical Q:\n", Q_empirical) # Optional: view raw Q

    # Apply Gaussian smoothing
    # mode='constant', cval=0.0 prevents edge effects by padding with zeros
    Q_smooth = gaussian_filter(Q_empirical, sigma=sigma, mode='constant', cval=0.0)
    print(Q_smooth.shape)
    # Renormalize the smoothed distribution
    q_sum = np.sum(Q_smooth)
    if q_sum > 1e-9:
        Q_smooth /= q_sum
    else:
        # Handle case where smoothing results in near-zero sum (unlikely)
        print("Warning: Smoothed Q sum is near zero.")
        # Reset to uniform? Or keep as is? Let's keep it near zero.
        pass

    print(f"Smoothed Q sum = {np.sum(Q_smooth):.4f}")
    # print("Smoothed Q:\n", Q_smooth) # Optional: view smoothed Q

    return Q_smooth, L_max, M_max

# --- Example Kappa Function (Unchanged) ---
def example_kappa_func(z1, z2, u, l, i, j):
    """
    Example placeholder kappa function.
    Transitions depend on the proportion of neighbour types.
    Control 'u' adjusts the base transition rates (example).
    Normalization is handled crudely here for demonstration.
    A real kappa function needs careful design for normalization.
    """
    k = l - i - j
    # Handle l=0 case first (no neighbours)
    if l == 0:
        base_stay_prob = 0.8
        if z1 == T: stay_prob = base_stay_prob + 0.1 * u
        elif z1 == H: stay_prob = base_stay_prob - 0.05 * u
        else: stay_prob = base_stay_prob - 0.05 * u
        stay_prob = np.clip(stay_prob, 0.1, 0.9)
        if z1 == z2: return stay_prob
        else: return (1.0 - stay_prob) / (NUM_STATES - 1)

    # Case l > 0
    prop_T, prop_H, prop_D = i / l, j / l, k / l
    base_T = 0.1 + 0.6 * prop_T
    base_H = 0.1 + 0.6 * prop_H
    base_D = 0.05 + 0.5 * prop_D
    influence_H_on_T = 0.15 * prop_H
    influence_T_on_H = 0.1 * prop_T
    influence_D_on_T = 0.05 * prop_D
    influence_D_on_H = 0.05 * prop_D
    control_factor_T = (1 + u) if u > 0 else 1.0
    control_factor_H = 1.0 / (1 + u) if u > 0 else 1.0
    raw_probs = {}
    prob_T = base_T - influence_H_on_T - influence_D_on_T
    if z1 != T: prob_T *= control_factor_T
    raw_probs[T] = max(0.0, prob_T)
    prob_H = base_H - influence_T_on_H - influence_D_on_H
    if z1 != H: prob_H *= control_factor_H
    raw_probs[H] = max(0.0, prob_H)
    prob_D = base_D
    raw_probs[D] = max(0.0, prob_D)
    total_prob = sum(raw_probs.values())
    if total_prob > 1e-9: return raw_probs[z2] / total_prob
    elif z1 == z2: return 1.0
    else: return 0.0

# --- Calculate Theta (Unchanged) ---
def calculate_theta(rho_matrix, Q, L_max, M_max):
    """
    Calculates theta_z = P(source node of random edge is in state z).
    rho_matrix: shape (L_max+1, NUM_STATES), rho_matrix[l, z] = rho_z^l
    Q: Joint degree distribution matrix, shape (L_max+1, M_max+1), Q[l, m] = P(in=l, out=m)
    """
    numerator = np.zeros(NUM_STATES)
    denominator = 0.0
    # Ensure loops use the actual dimensions of Q
    L_max_q, M_max_q = Q.shape[0] - 1, Q.shape[1] - 1

    for l in range(L_max_q + 1):
        for m in range(M_max_q + 1):
            if Q[l, m] == 0: continue
            term_den = m * Q[l, m]
            denominator += term_den
            # Ensure rho_matrix access is within bounds
            if l <= L_max:
                 for z in STATES:
                     numerator[z] += term_den * rho_matrix[l, z]
            # Else: If Q has degrees higher than rho_matrix dimension L_max,
            # those nodes don't contribute state info to theta calculation here.
            # This assumes rho_matrix covers all degrees present in Q.

    if denominator < 1e-9:
        print("Warning: Average out-degree is near zero. Returning uniform theta.")
        return np.ones(NUM_STATES) / NUM_STATES

    thetas = numerator / denominator
    thetas = thetas / np.sum(thetas)
    return thetas

# --- ODE System Definition (Small modification for L_max/M_max args) ---
def llm_ode_full(rho_vec, t, Q, u, kappa_func):
    """
    Defines the full ODE system d(rho^l)/dt for all l.
    rho_vec: flattened state vector, size (L_max+1) * NUM_STATES
             where L_max is determined by Q's shape.
    t: time
    Q: Joint degree distribution matrix (shape determines L_max, M_max)
    u: control parameter
    kappa_func: function kappa(z1, z2, u, l, i, j)
    """
    # Determine L_max, M_max from Q's shape
    L_max = Q.shape[0] - 1
    M_max = Q.shape[1] - 1
    expected_rho_size = (L_max + 1) * NUM_STATES

    # Check if rho_vec size matches Q dimensions
    if len(rho_vec) != expected_rho_size:
         raise ValueError(f"rho_vec size {len(rho_vec)} does not match expected size {expected_rho_size} based on Q shape {Q.shape}")

    # Reshape and normalize rho_vec
    rho_matrix = rho_vec.reshape((L_max + 1, NUM_STATES))
    for l in range(L_max + 1):
        row_sum = np.sum(rho_matrix[l, :])
        if row_sum > 1e-9: rho_matrix[l, :] /= row_sum
        else: rho_matrix[l, :] = 1.0 / NUM_STATES

    # Calculate current thetas based on the full state rho_matrix and Q
    theta_T, theta_H, theta_D = calculate_theta(rho_matrix, Q, L_max, M_max)

    # Initialize derivative matrix
    d_rho_matrix_dt = np.zeros_like(rho_matrix)

    # Calculate derivatives for each degree l
    for l in range(L_max + 1):
        rho_l = rho_matrix[l, :] # Current state for degree l

        # Calculate G^l matrix (transitions for degree l)
        G_l = np.zeros((NUM_STATES, NUM_STATES))
        # --- G calculation loop (identical to previous version) ---
        for z1 in STATES:
            for z2 in STATES:
                if z1 == z2: continue
                sum_val = 0.0
                for i in range(l + 1): # Truthful neighbours
                    for j in range(l - i + 1): # Hallucinating neighbours
                        k = l - i - j # 'Don't know' neighbours
                        if k < 0: continue
                        try: multinomial_coeff = comb(l, i) * comb(l - i, j)
                        except ValueError: multinomial_coeff = 0
                        if multinomial_coeff == 0: continue
                        kappa_val = kappa_func(z1, z2, u, l, i, j)
                        term_prob = 1.0
                        term_prob *= np.power(theta_T, i) if theta_T > 1e-9 or i == 0 else 0.0
                        term_prob *= np.power(theta_H, j) if theta_H > 1e-9 or j == 0 else 0.0
                        term_prob *= np.power(theta_D, k) if theta_D > 1e-9 or k == 0 else 0.0
                        if (theta_T < 1e-9 and i > 0) or \
                           (theta_H < 1e-9 and j > 0) or \
                           (theta_D < 1e-9 and k > 0):
                           term_prob = 0.0
                        sum_val += kappa_val * multinomial_coeff * term_prob
                G_l[z1, z2] = sum_val
        # --- End G calculation ---

        # Calculate F^l_odeint matrix
        F_l_odeint = np.zeros((NUM_STATES, NUM_STATES))
        # --- F calculation loop (identical to previous version) ---
        for z1 in STATES:
            sum_G_z1_zprime = 0.0
            for z_prime in STATES:
                if z1 == z_prime: continue
                F_l_odeint[z_prime, z1] = G_l[z1, z_prime]
                sum_G_z1_zprime += G_l[z1, z_prime]
            F_l_odeint[z1, z1] = -sum_G_z1_zprime
        # --- End F calculation ---

        # Calculate derivative for degree l
        d_rho_l_dt = F_l_odeint @ rho_l
        d_rho_matrix_dt[l, :] = d_rho_l_dt

    # Flatten the derivative matrix back into a vector
    d_rho_vec_dt = d_rho_matrix_dt.flatten()
    return d_rho_vec_dt

# --- Simulation Parameters ---
U_control = 0.1  # Example control parameter (-1 < u < inf)
Smoothing_Sigma = 0.8 # Sigma for Gaussian smoothing of Q

# --- Example Adjacency Matrix ---
# Replace this with your actual adjacency matrix (N x N)
# A[i, j] = 1 means edge j -> i
# Example: 10 nodes, some connections
np.random.seed(42) # for reproducibility
N_nodes = 20
example_A = np.random.rand(N_nodes, N_nodes) < 0.15 # ~15% connection probability
np.fill_diagonal(example_A, 0) # No self-loops

print(f"Using example Adjacency Matrix (N={N_nodes})")
# print(example_A.astype(int))

# --- Compute and Smooth Q ---
Q, L_max, M_max = compute_and_smooth_q(example_A, sigma=Smoothing_Sigma)

# --- Initial State rho0 ---
# Define initial state rho^l for each degree l up to L_max
rho0_matrix = np.zeros((L_max + 1, NUM_STATES))
rho0_matrix[:, T] = 0.1  # 10% Truthful initially
rho0_matrix[:, H] = 0.8  # 80% Hallucinating initially
rho0_matrix[:, D] = 0.1  # 10% Don't Know initially

# Ensure rows sum to 1
for l in range(L_max + 1):
     rho0_matrix[l, :] /= np.sum(rho0_matrix[l, :])

# Flatten for odeint
rho0_vec = rho0_matrix.flatten()

# --- Time Span ---
t_span = np.linspace(0, 200, 250) # Simulate for longer time, more points

# --- Run Simulation ---
print(f"Starting simulation for L_max={L_max}, M_max={M_max}, u={U_control}, sigma={Smoothing_Sigma}")
print(f"Initial state rho0 (example): T={rho0_matrix[0,T]:.2f}, H={rho0_matrix[0,H]:.2f}, D={rho0_matrix[0,D]:.2f}")

# Note: odeint args now only pass Q, u, kappa_func. L_max/M_max derived inside.
sol_vec = odeint(llm_ode_full, rho0_vec, t_span, args=(Q, U_control, example_kappa_func))

print("Simulation finished.")

# --- Process Results ---
# Reshape solution vector back into matrix form over time
# Shape: (num_time_points, L_max+1, NUM_STATES)
sol_matrix = sol_vec.reshape((len(t_span), L_max + 1, NUM_STATES))

# Calculate the marginal probability of each in-degree l: P(l) = sum_m Q(l, m)
P_l = np.sum(Q, axis=1) # Sum over columns (m)

# Calculate the average state distribution over time
# rho_avg_z(t) = sum_l P(l) * rho_z^l(t)
# Ensure P_l aligns with sol_matrix dimensions if Q had higher L_max
P_l_broadcast = P_l[:L_max+1].reshape(1, L_max + 1, 1) # Reshape for broadcasting

rho_avg = np.sum(P_l_broadcast * sol_matrix, axis=1) # Sum over l dimension
rho_avg_T = rho_avg[:, T]
rho_avg_H = rho_avg[:, H]
rho_avg_D = rho_avg[:, D]

print(f"Final average state rho_avg({t_span[-1]}): T={rho_avg_T[-1]:.3f}, H={rho_avg_H[-1]:.3f}, D={rho_avg_D[-1]:.3f}")
print(f"Final average state sum = {np.sum(rho_avg[-1, :]):.3f}") # Should be close to 1

# --- Plot Results ---
plt.figure(figsize=(12, 7))

# Plot average state evolution
plt.plot(t_span, rho_avg_T, label='Avg. Truthful ($\\bar{\\rho}_T$)', color='green', linewidth=2)
plt.plot(t_span, rho_avg_H, label='Avg. Hallucinating ($\\bar{\\rho}_H$)', color='red', linewidth=2)
plt.plot(t_span, rho_avg_D, label='Avg. Does not know ($\\bar{\\rho}_D$)', color='gray', linewidth=2)

plt.title(f'LLM Latent State Evolution (N={N_nodes}, Control u={U_control}, Q Smooth $\sigma$={Smoothing_Sigma})')
plt.xlabel('Time (t)')
plt.ylabel('Proportion')
plt.legend(loc='best')
plt.grid(True)
plt.ylim([-0.05, 1.05])
plt.tight_layout()
plt.show()

