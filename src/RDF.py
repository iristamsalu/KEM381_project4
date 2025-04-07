import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import os
import argparse

# Read the XYZ file
def read_xyz_frames(filename):
    with open(filename, 'r') as f:
        while True:
            header = f.readline()
            if not header:
                break
            try:
                N = int(header.strip())
            except ValueError:
                print(f"Warning: Skipping invalid header line: {header.strip()}")
                # Skip to next potential header
                continue 
            # Read comment line
            comment = f.readline() 
            # Check if EOF reached after reading comment
            if not comment: 
                 break

            frame_data = []
            valid_frame = True
            for i in range(N):
                line = f.readline()
                # Check if end of the file reached unexpectedly within frame
                if not line: 
                    print(f"Warning: Unexpected end of file while reading frame.")
                    valid_frame = False
                    break
                parts = line.split()
                try:
                    # Expecting format like: Particle X Y Z
                    frame_data.append([float(val) for val in parts[1:]])
                except (IndexError, ValueError):
                     print(f"Warning: Skipping invalid data line in frame: {line.strip()}")
                     valid_frame = False
                     # Read remaining lines of this supposed frame to advance file pointer
                     for _ in range(N - 1 - i):
                         f.readline()
                    # Break from inner loop
                     break 

            if valid_frame and len(frame_data) == N:
                 yield np.array(frame_data), N
            elif not valid_frame:
                 print("Skipping potentially corrupted frame.")


@jit(nopython=True, parallel=True)
def compute_rdf_parallel(positions, N, L, cutoff, bins, dim):
    distances = np.empty((N * (N - 1)) // 2)
    idx = 0
    for i in prange(N):
        for j in range(i + 1, N):
            rij = positions[j] - positions[i]
            for k in range(dim):
                 if L[k] > 0:
                      rij[k] -= L[k] * np.round(rij[k] / L[k])
            dist = np.sqrt(np.sum(rij[:dim]**2))
            if idx < distances.shape[0]:
                 distances[idx] = dist
            idx += 1

    actual_distances = distances[:idx]
    # Histogram calculation
    # Ensure the range starts exactly at 0 for this correction method
    hist, bin_edges = np.histogram(actual_distances, bins=bins, range=(0.0, cutoff))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dr = bin_edges[1] - bin_edges[0]
    g_r = np.zeros_like(bin_centers)

    if dim == 2:
        area = L[0] * L[1]
        rho = N / area
        for i in range(bins):
            # Correction for the first bin to avoid the high peak at small r
            if i == 0:
                 # Explicitly set g(r) for the first bin to 0
                 g_r[i] = 0.0 
                 # Skip the rest of the calculation for this bin
                 continue 

            shell_area = np.pi * (bin_edges[i+1]**2 - bin_edges[i]**2)
            # Normalization
            norm = rho * shell_area * N / 2.0 
            if norm > 1e-9:
                 g_r[i] = hist[i] / norm

    else: # dim == 3
         volume = L[0] * L[1] * L[2]
         rho = N / volume
         for i in range(bins):
             # Correction for the first bin to avoid high peak at low r
             if i == 0:
                  # Set g(r) for the first bin to 0
                  g_r[i] = 0.0 
                  # Skip the rest of the calculation for this bin
                  continue 
             shell_volume = 4.0/3.0 * np.pi * (bin_edges[i+1]**3 - bin_edges[i]**3)
             # Normalization
             norm = rho * shell_volume * N / 2.0 
             if norm > 1e-9:
                  g_r[i] = hist[i] / norm
             # else: g_r[i] remains 0

    return bin_centers, g_r

def main(filename, density, start_frame, bins, dim, output_dir="output"):
    print(f"Starting RDF calculation for {filename}")
    print(f"Parameters: density={density}, start_frame={start_frame}, bins={bins}, dim={dim}")

    frame_generator = read_xyz_frames(filename)
    rdf_sum = None
    bin_centers = None
    frame_count = 0
    processed_frame_count = 0
    # Initialize particle count
    N = -1 

    for current_frame_index, (frame_data, current_N) in enumerate(frame_generator):
        if N == -1:
             N = current_N
             print(f"Detected N = {N} particles.")
             # Calculate box size based on the first frame's N
             if dim == 2:
                 box_size_xy = (N / density) ** 0.5
                 # Lx, Ly, Lz=0
                 box_size = np.array([box_size_xy, box_size_xy, 0.0]) 
                 cutoff = box_size_xy / 2.0
                 print(f"Calculated 2D Box Size: Lx=Ly={box_size_xy:.4f}, Cutoff: {cutoff:.4f}")
             else: # dim == 3
                 box_size_xyz = (N / density) ** (1.0 / 3.0)
                 # Lx=Ly=Lz
                 box_size = np.array([box_size_xyz] * 3) 
                 cutoff = box_size_xyz / 2.0
                 print(f"Calculated 3D Box Size: Lx=Ly=Lz={box_size_xyz:.4f}, Cutoff: {cutoff:.4f}")
             if cutoff <= 0:
                  print("Error: Calculated cutoff radius is zero or negative. Check density and N.")
                  return
             if bins <= 0:
                  print("Error: Number of bins must be positive.")
                  return

        elif current_N != N:
            print(f"Warning: Frame {current_frame_index} has {current_N} particles, expected {N}. Skipping frame.")
            continue

        if current_frame_index < start_frame:
            # Skip frames before start_frame
            continue 

        # Ensure positions are correctly sliced for the given dimension
        positions = frame_data[:, :dim]

        # Call the compute function
        current_bin_centers, g_r = compute_rdf_parallel(positions, N, box_size, cutoff, bins, dim)

        if rdf_sum is None:
            rdf_sum = g_r
            # Store bin centers from the first processed frame
            bin_centers = current_bin_centers 
        else:
            # Ensure g_r has the same shape before adding
            if g_r.shape == rdf_sum.shape:
                 rdf_sum += g_r
            else:
                 print(f"Warning: Shape mismatch in g_r for frame {current_frame_index}. Expected {rdf_sum.shape}, got {g_r.shape}. Skipping frame.")
                 # Skip this frame's contribution
                 continue 

        processed_frame_count += 1
        # Print progress
        if processed_frame_count % 1000 == 0: 
             print(f"Processed {processed_frame_count} frames...")


    if processed_frame_count == 0:
        print("Error: No frames were processed. Check start_frame or file content.")
        return

    if rdf_sum is None or bin_centers is None:
         print("Error: RDF calculation failed (rdf_sum or bin_centers is None).")
         return

    g_r_avg = rdf_sum / processed_frame_count

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, g_r_avg, color='crimson', linestyle='-')
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title(f"Radial Distribution Function")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, cutoff)
    # Keep bottom at 0
    plt.ylim(bottom=0) 
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, "rdf_plot.png")
    data_filename = os.path.join(output_dir, "rdf_data.dat")

    plt.savefig(plot_filename)
    print(f"RDF plot saved to {plot_filename}")

    # Save data
    try:
        np.savetxt(data_filename, np.column_stack((bin_centers, g_r_avg)), header="r g(r)", fmt="%.6f")
        print(f"RDF data saved to {data_filename}")
    except Exception as e:
        print(f"Error saving RDF data: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RDF from XYZ file.")
    parser.add_argument("filename", type=str, help="Path to the input XYZ file")
    parser.add_argument("--density", type=float, required=True, help="System density (required)")
    parser.add_argument("--dim", type=int, choices=[2, 3], required=True, help="Dimension of the simulation (2 or 3, required)")
    parser.add_argument("--start", type=int, default=0, help="Frame index to start averaging from (0-based, default: 0)")
    parser.add_argument("--bins", type=int, default=100, help="Number of RDF histogram bins (default: 100)")
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"Error: Input file not found: {args.filename}")
    else:
        main(args.filename, args.density, args.start, args.bins, args.dim)