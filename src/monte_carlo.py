from numba import jit, prange
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Monte Carlo simulation of Lennard-Jones particles')
    parser.add_argument('--dimension', '-d', type=int, default=2, choices=[2, 3],
                        help='System dimensionality (2 or 3)')
    parser.add_argument('--particles', '-n', type=int, default=500,
                        help='Number of particles')
    parser.add_argument('--density', '-rho', type=float, default=0.94,
                        help='Number density')
    parser.add_argument('--temperature', '-t', type=float, default=1.52,
                        help='Reduced temperature')
    parser.add_argument('--steps', '-s', type=int, default=1000000,
                        help='Number of Monte Carlo steps')
    parser.add_argument('--max_displacement', '-md', type=float, default=0.08,
                        help='Maximum displacement')
    parser.add_argument('--cutoff', '-c', type=float, default=2.5,
                        help='Cutoff distance in units of sigma')
    parser.add_argument('--seed', '-seed', type=int, default=16,
                        help='Random seed for reproducibility')
    parser.add_argument('--sigma', '-sig', type=float, default=1.0,
                        help='Lennard-Jones sigma')
    parser.add_argument('--epsilon', '-eps', type=float, default=1.0,
                        help='Lennard-Jones epsilon')


    return parser.parse_args()


# Replace the parameter definitions with:
args = parse_arguments()
N = args.particles
rho = args.density
dim = args.dimension
T = args.temperature
sigma = args.sigma
epsilon = args.epsilon
cutoff = args.cutoff * sigma # Lennard-Jones cutoff in units of sigma
n_steps = args.steps
max_disp = args.max_displacement

# Define output directory and create it if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Box length
L = (N / rho) ** (1 / dim)

# Set random seed for reproducibility
np.random.seed(args.seed)

# Initialize positions array
positions = np.zeros((N, dim))
positions[0] = L * np.random.rand(dim)  # Place first particle

# Place remaining particles
for i in range(1, N):  # Start from 1 since particle 0 is already placed
    attempts = 0
    while attempts < 2000:
        # Generate a random position for particle i
        positions[i] = L * np.random.rand(dim)
        # Calculate distances to all previous particles
        distances = np.linalg.norm(positions[:i] - positions[i], axis=1)
        # Check if all distances are greater than sigma
        if np.all(distances > sigma):
            break
        attempts += 1

    if attempts == 1000:
        raise Exception(f"Could not find valid position for particle {i}")



# Compute the Lennard-Jones potential
@jit(nopython=True)
def lennard_jones_potential(r, cutoff, sigma, epsilon):
    if r < cutoff:
        inv_r6 = (sigma / r) ** 6
        inv_r12 = inv_r6 ** 2
        return 4 * epsilon * (inv_r12 - inv_r6)
    else:
        return 0.0


# Compute total potential energy
@jit(nopython=True)
def total_energy(positions, N, L, cutoff, sigma, epsilon):
    energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rij = positions[j] - positions[i]
            rij -= L * np.round(rij / L)  # Apply minimum image convention
            r = np.linalg.norm(rij)
            energy += lennard_jones_potential(r, cutoff, sigma, epsilon)
    return energy


@jit(nopython=True)
def compute_energy_diff(i, old_pos, new_pos, positions, N, L, cutoff, sigma, epsilon):
    delta_E = 0.0
    for j in range(N):
        if j != i:
            rij_old = positions[j] - old_pos
            rij_old -= L * np.round(rij_old / L)
            r_old = np.linalg.norm(rij_old)

            rij_new = positions[j] - new_pos
            rij_new -= L * np.round(rij_new / L)
            r_new = np.linalg.norm(rij_new)

            delta_E += (lennard_jones_potential(r_new, cutoff, sigma, epsilon) -
                       lennard_jones_potential(r_old, cutoff, sigma, epsilon))
    return delta_E


@jit(nopython=True)
def compute_msd(positions_history, N, L, dim):
    n_steps = len(positions_history)
    msd = np.zeros(n_steps)
    initial_positions = positions_history[0]

    for step in range(n_steps):
        current_positions = positions_history[step]
        displacement = current_positions - initial_positions

        # Apply PBC to displacement
        displacement -= L * np.round(displacement / L)

        # Calculate squared displacement for each particle
        squared_displacement = np.sum(displacement ** 2, axis=1)

        # Average over all particles
        msd[step] = np.mean(squared_displacement)

    return msd


# Track energy and displacement statistics
@jit(nopython=True)
def run_simulation(positions, N, L, T, cutoff, sigma, epsilon, n_steps, max_disp, dim):
    energy = total_energy(positions, N, L, cutoff, sigma, epsilon)
    accepted_moves = 0
    energies = np.zeros(n_steps)
    displacements = np.zeros(n_steps)
    positions_history = np.zeros((n_steps, N, dim))  # Store all positions

    for step in range(n_steps):
        i = np.random.randint(N)
        old_pos = positions[i].copy()
        displacement = max_disp * (2 * np.random.rand(dim) - 1)
        new_pos = old_pos + displacement
        new_pos = new_pos % L

        delta_E = compute_energy_diff(i, old_pos, new_pos, positions, N, L, cutoff, sigma, epsilon)

        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            positions[i] = new_pos
            energy += delta_E
            accepted_moves += 1
            displacements[step] = np.linalg.norm(displacement)

        energies[step] = energy
        positions_history[step] = positions.copy()

    return positions, energies, accepted_moves, displacements, positions_history


print("Starting simulation...")
print(f"Number of particles: {N}, Box length: {L:.2f}, Density: {rho:.2f}, Temperature: {T:.2f}, Number of steps: {n_steps}, Max displacement: {max_disp:.3f}, Dimensionality: {dim}")
start_time = time.time()
# Run the simulation
positions, energies, accepted_moves, displacements, positions_history = run_simulation(
    positions, N, L, T, cutoff, sigma, epsilon, n_steps, max_disp, dim
)
end_time = time.time()

print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
# simulation time per step
simulation_time_per_step = (end_time - start_time) / n_steps
print(f"Average time per step: {simulation_time_per_step:.6f} seconds.")

# Compute acceptance ratio
# Compute statistics and plots after simulation
acceptance_ratio = accepted_moves / n_steps
print(f"Acceptance Ratio: {acceptance_ratio:.3f}")
print(f"Final Energy per Particle: {energies[-1] / N:.3f}")


print("Computing MSD...")
msd = compute_msd(positions_history, N, L, dim)

plt.figure()
plt.plot(range(n_steps), msd, label='MSD')
plt.xlabel("MC Steps")
plt.ylabel("Mean Square Displacement")
plt.legend()
plt.title("Mean Square Displacement")
plt.savefig(os.path.join(output_dir, "msd_MC.png"))



@jit(nopython=True)
def autocorrelation(data, max_lag):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)

    if var == 0:
        return np.zeros(max_lag)

    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        # Normalize by number of pairs at each lag
        n_pairs = n - lag
        if n_pairs < 1:
            break

        sum_corr = 0.0
        for i in range(n_pairs):
            sum_corr += (data[i] - mean) * (data[i + lag] - mean)

        acf[lag] = sum_corr / (n_pairs * var)

    return acf


@jit(nopython=True)
def compute_integrated_correlation_time(acf):
    """Compute integrated correlation time using windowing method."""
    n = len(acf)
    tau_int = np.zeros(n)

    # Compute running sum of autocorrelation
    for window in range(1, n):
        tau_int[window] = 0.5 + np.sum(acf[1:window])

        # Check for convergence using window method
        if window > 4 * tau_int[window]:
            return tau_int[:window], window

    return tau_int, n



# Usage:
print("Computing autocorrelation...")
max_lag = min(n_steps // 10, 10000)  # Limit max_lag to 10% of n_steps or 10000
auto_corr = autocorrelation(energies, max_lag)

tau_int, window_size = compute_integrated_correlation_time(auto_corr)
# interpretation
n_independent = n_steps / (2 * tau_int[window_size-1])
print(f"Number of independent samples: {n_independent:.0f}")


# Plot both autocorrelation and integrated correlation time
plt.figure(figsize=(10, 5))

# Subplot 1: Autocorrelation
plt.subplot(1, 2, 1)
plt.plot(range(len(auto_corr)), auto_corr, label='ACF')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Energy Autocorrelation")

# Subplot 2: Integrated correlation time
plt.subplot(1, 2, 2)
plt.plot(range(len(tau_int)), tau_int, label='τ_int')
plt.axvline(x=window_size, color='r', linestyle='--', label='Window')
plt.xlabel("Window Size")
plt.ylabel("τ_int")
plt.title(f"Integrated Correlation Time\nτ_int = {tau_int[window_size-1]:.1f}")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_analysis_MC.png"))


# Plot energy evolution
plt.figure()
plt.plot(range(n_steps), energies, label='Total Energy')
plt.xlabel("MC Steps")
plt.ylabel("Energy")
plt.legend()
plt.title("Energy Evolution")
plt.savefig(os.path.join(output_dir, "energy_evolution_MC.png"))


@jit(nopython=True, parallel=True)
def compute_rdf_parallel(positions_snapshots, N, L, cutoff, dim, bins=100):
    n_snapshots = len(positions_snapshots)
    rdf_accum = np.zeros(bins)
    bin_edges = np.linspace(0, cutoff, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dr = bin_edges[1] - bin_edges[0]

    # Compute normalization factors once
    rho = N / (L ** dim)
    if dim == 2:
        norm = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2) * rho * N
    else:  # dim == 3
        norm = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3) * rho * N

    for snapshot in positions_snapshots:
        hist = np.zeros(bins)
        for i in prange(N):
            for j in range(i + 1, N):
                rij = snapshot[j] - snapshot[i]
                # Apply minimum image convention
                for d in range(dim):
                    rij[d] -= L * round(rij[d] / L)
                r = np.sqrt(np.sum(rij * rij))
                if r < cutoff:
                    bin_idx = int(r / dr)
                    if bin_idx < bins:
                        hist[bin_idx] += 2  # Count each pair twice

        # Avoid division by zero if norm is zero for some bins
        safe_norm = np.where(norm == 0, 1, norm) # Replace 0 with 1 in norm array where it's 0
        rdf_accum += hist / safe_norm


    rdf_avg = rdf_accum / n_snapshots
    return bin_centers, rdf_avg



print("Computing radial distribution function...")
rdf_cutoff = min(L/2, 7.0)
# take the last 20% of the snapshots for RDF calculation
positions_history_rdf = positions_history[int(n_steps * 0.8):]
r_vals, rdf_vals = compute_rdf_parallel(positions_history_rdf, N, L, rdf_cutoff, dim)
plt.figure()
plt.plot(r_vals, rdf_vals, label='Radial Distribution Function')
plt.xlabel("r")
plt.ylabel("g(r)")
plt.legend()
plt.title("Radial Distribution Function (RDF)")
plt.savefig(os.path.join(output_dir, "rdf_MC.png"))

# Plot histogram of displacements
plt.figure()
plt.hist(displacements[displacements > 0], bins=70, alpha=0.7, color='g')
plt.xlabel("Displacement Distance")
plt.ylabel("Frequency")
plt.title("Histogram of Accepted Particle Displacements")
plt.savefig(os.path.join(output_dir, "displacement_hist_MC.png"))

# Plot final positions
def plot_final_positions(positions, L, dim, output_dir):
    if dim == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(positions[:, 0], positions[:, 1], s=30, alpha=0.7)
        plt.xlim(0, L)
        plt.ylim(0, L)
        plt.title("Final Particle Configuration (2D)")
        plt.xlabel("x")
        plt.ylabel("y")
    else:  # dim == 3
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=30, alpha=0.7)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_zlim(0, L)
        ax.set_title("Final Particle Configuration (3D)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    plt.savefig(os.path.join(output_dir, f"final_positions_{dim}d_MC.png"))


plot_final_positions(positions, L, dim, output_dir)


print(f"All plots saved to the '{output_dir}' directory with the '_MC' tag.")