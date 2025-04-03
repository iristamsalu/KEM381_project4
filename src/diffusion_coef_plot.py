import numpy as np
import time as timer
import matplotlib.pyplot as plt

def compute_diffusion_coefficient(msd, time, dimensions):
    """
    Compute the diffusion coefficient from MSD vs. time using linear regression.
    Fits the latter half of the data by default.

    Args:
        msd (np.array): Array of MSD values.
        time (np.array): Array of corresponding time values.
        dimensions (int): Number of dimensions (2 or 3).

    Returns:
        float: Estimated diffusion coefficient, or np.nan if calculation fails.
    """
    if len(time) < 2 or len(msd) < 2 or dimensions not in [1, 2, 3]:
        print("Warning: Not enough data points or invalid dimensions to compute diffusion coefficient.")
        return np.nan

    # Fit MSD = (2 * d * D) * time + C
    # We want the slope = 2 * d * D
    try:
        # Fit the linear part, often at later times. Heuristic: use from the one third.
        start_fit_index = len(time) // 3
        # Ensure we have at least 2 points for the fit
        if start_fit_index >= len(time) - 1:
            start_fit_index = 0 # Use all data if second half is too small

        fit_time = time[start_fit_index:]
        fit_msd = msd[start_fit_index:]

        if len(fit_time) < 2:
             print("Warning: Not enough data points in the selected range for fitting D. Using all data.")
             fit_time = time
             fit_msd = msd
             if len(fit_time) < 2: # Still not enough
                  print("Error: Cannot perform linear fit with less than 2 points.")
                  return np.nan

        print(f"Fitting diffusion coefficient using data from time={fit_time[0]:.4f} onwards ({len(fit_time)} points).")
        # Perform linear regression: msd = slope * time + intercept
        slope, intercept = np.polyfit(fit_time, fit_msd, 1)

        if slope < 0:
            print(f"Warning: Calculated slope of MSD vs time is negative ({slope:.2e}). This might indicate issues with simulation or analysis.")

        # D = slope / (2 * d)
        diffusion_coef = slope / (2 * dimensions)
        return diffusion_coef

    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Warning: Linear fit for diffusion coefficient failed: {e}")
        return np.nan

def process_trajectory_for_msd(filename, dt, skip_interval=1, start_frame=0):
    """
    Reads an XYZ file frame by frame, calculates MSD relative to the specified start frame.

    Args:
        filename (str): Path to the XYZ file.
        dt (float): Timestep duration between frames in the original file.
        skip_interval (int): Interval for skipping frames (e.g., 1 processes every frame, 5 processes every 5th frame).
        start_frame (int): The frame number (in the *processed* sequence) from which to start the MSD calculation.

    Returns:
        tuple: (msd_array, time_array, dimensions, processed_frame_count)
               Returns (None, None, None, None) if an error occurs.
    """
    msd_values = []
    times = []
    first_positions = None
    n_particles = 0
    dimensions = 3
    # Flag to indicate if the start_frame has been found
    first_frame_found = False
    current_time = 0.0 # Initialize current_time here
    current_msd = 0.0 # Initialize current_msd here

    try:
        with open(filename, 'r') as f:
            line_num = 0
            frame_count_total = 0
            frame_count_processed = 0
            start_time = timer.time()

            print(f"Processing trajectory file: {filename}")
            print(f"Timestep (dt): {dt}, Skip Interval: {skip_interval}, Start Frame: {start_frame}")

            while True:
                # Read header lines for a frame
                line1 = f.readline()
                if not line1:
                    break

                try:
                    current_n_particles = int(line1.strip())
                    if frame_count_total == 0:
                        n_particles = current_n_particles
                        print(f"Number of particles per frame: {n_particles}")
                    elif current_n_particles != n_particles:
                        print(f"Warning: Number of particles changed from {n_particles} to {current_n_particles} at frame {frame_count_total}. Aborting.")
                        return None, None, None, None
                except ValueError:
                    print(f"Error: Could not read number of particles at line {line_num + 1}. Content: '{line1.strip()}'")
                    return None, None, None, None
                line_num += 1

                # Read comment line (can be any comment)
                f.readline()
                line_num += 1

                # Decide whether to process or skip this frame
                if frame_count_total % skip_interval == 0:
                    # Read particle coordinates for this frame
                    current_positions_raw = np.zeros((n_particles, 3))
                    try:
                        for i in range(n_particles):
                            line = f.readline()
                            if not line:
                                raise EOFError(f"Unexpected end of file while reading particle data for frame {frame_count_total}.")
                            line_num += 1
                            parts = line.split()
                            offset = 0
                            if not parts[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                                offset = 1
                            if len(parts) < offset + 3:
                                raise ValueError(f"Insufficient coordinate data on line {line_num}. Content: '{line.strip()}'")

                            x = float(parts[offset])
                            y = float(parts[offset+1])
                            z = float(parts[offset+2])
                            current_positions_raw[i, 0] = x
                            current_positions_raw[i, 1] = y
                            current_positions_raw[i, 2] = z
                    except (ValueError, IndexError) as e:
                         print(f"Error parsing coordinates at line {line_num}: {e}. Content: '{line.strip()}'")
                         return None, None, None, None
                    except EOFError as e:
                         print(e)
                         return None, None, None, None
                    # End reading particle coordinates

                    # Process the frame
                    if frame_count_processed < start_frame:
                        # Skip until the start frame
                        pass  
                        #print(f"Skipping frame {frame_count_processed} (Original frame: {frame_count_total})")
                    elif not first_frame_found:
                        # This is the start frame
                        first_positions_raw = current_positions_raw.copy()
                        # Determine dimensions
                        if np.all(first_positions_raw[:, 2] == 0.0):
                             dimensions = 2
                             print("Detected 2D system based on the first processed frame (Z=0).")
                             first_positions = first_positions_raw[:, :2] # Store only X, Y
                        else:
                             dimensions = 3
                             print("Assuming 3D system.")
                             first_positions = first_positions_raw # Store X, Y, Z
                        # Append 0 for MSD at t=0
                        msd_values.append(0.0)
                        times.append(0.0)
                        first_frame_found = True # Flag to ensure this block is only executed once
                        print(f"Starting MSD calculation from frame {frame_count_processed} (Original frame: {frame_count_total})")
                    else:
                        # System is past the start frame, Calculate MSD

                        # Prepare current positions based on detected dimensions
                        if dimensions == 2:
                            current_positions = current_positions_raw[:, :2]
                        else:
                            current_positions = current_positions_raw

                        # Calculate displacement relative to the start frame
                        displacement = current_positions - first_positions
                        squared_displacement = np.sum(displacement**2, axis=1)
                        current_msd = np.mean(squared_displacement)

                        msd_values.append(current_msd)
                        current_time = (frame_count_processed - start_frame) * dt * skip_interval
                        times.append(current_time)

                    frame_count_processed += 1

                    # Progress indicator
                    if frame_count_processed % 10000 == 0:
                         elapsed = timer.time() - start_time
                         print(f"... Processed frame {frame_count_processed} (Original frame: {frame_count_total}) Time: {current_time:.4f} MSD: {current_msd:.4e} Elapsed: {elapsed:.2f}s")

                else:
                    # Skip this frame's coordinate lines
                    try:
                        for _ in range(n_particles):
                            line = f.readline()
                            if not line:
                                raise EOFError(f"Unexpected end of file while reading particle data for frame {frame_count_total}.")
                            line_num += 1
                    except EOFError as e:
                         print(f"Warning: {e} File might end unexpectedly.")
                         break

                # End frame processing/skipping
                frame_count_total += 1

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred around line {line_num}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # Final checks and return
    if n_particles == 0 or first_positions is None:
        print("Error: Could not read any particle data or process any frames.")
        return None, None, None, None
    if len(times) < 2: # Need at least 2 points after start_frame
        print("Warning: Less than two frames were processed *after* start_frame. Cannot calculate diffusion coefficient.")
        # Still return what we have
        pass

    end_time = timer.time()
    print(f"\nFinished processing.")
    print(f"Read {frame_count_total} total frames from the file.")
    print(f"Processed {frame_count_processed} frames after applying skip_interval={skip_interval}.")
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")

    return np.array(msd_values), np.array(times), dimensions, frame_count_processed

# ==============================================
# --- Main Execution ---
# ==============================================

# Parameters
input_file = "2D_trajectory.xyz"
dt = 0.0001
skip_interval = 1
# Specify the starting frame here in processed frame count (frames that are in .XYZ)
start_frame = 1

print(f"Starting MSD analysis for '{input_file}'...")
print("-" * 30)

# Process the trajectory
msd, time, dimensions, num_processed_frames = process_trajectory_for_msd(input_file, dt, skip_interval, start_frame)

print("-" * 30)

# Analyze results
if msd is not None and time is not None and num_processed_frames > 0:
    sim_time_processed = time[-1] if len(time) > 0 else 0.0

    print("\n--- Analysis Results ---")
    print(f"System Dimensions: {dimensions}D")
    print(f"Number of processed frames (after start frame): {num_processed_frames}")
    print(f"Total time covered by processed frames (after start frame): {sim_time_processed:.5f} (simulation time units)")

    # Calculate Diffusion Coefficient
    D = compute_diffusion_coefficient(msd, time, dimensions)

    if not np.isnan(D):
        print(f"Estimated Diffusion Coefficient (D): {D:.5e}")
    else:
        print("Diffusion Coefficient could not be reliably estimated.")

    # Plotting
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(time, msd, '-', markersize=4, linewidth=1.5, label=f'MSD (from frame {start_frame})')
        plt.xlabel(f"Time (Simulation Units, from frame {start_frame})")
        plt.ylabel("Mean Squared Displacement (MSD)")
        plt.title(f"MSD vs. Time for {input_file}\n(dt={dt}, skip={skip_interval}, Start Frame={start_frame})")

        # Linear fit
        if not np.isnan(D) and len(time) > 1:
            start_fit_index = len(time) // 3
            if start_fit_index >= len(time) - 1: start_fit_index = 0
            fit_time = time[start_fit_index:]
            if len(fit_time) >= 2:
                slope = 2 * dimensions * D
                intercept = np.mean(msd[start_fit_index:]) - slope * np.mean(fit_time)
                plt.plot(fit_time, slope * fit_time + intercept, '--', color='red', linewidth=2, label=f'Linear Fit (D={D:.2e})')

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plot_filename = f"msd_plot_{input_file}.png"
        plt.savefig(plot_filename)
        print(f"MSD plot saved to {plot_filename}")

    except ImportError:
        print("\nMatplotlib not found. Cannot generate plot.")
    except Exception as e:
        print(f"\nAn error occurred during plotting: {e}")

else:
    print("\nAnalysis could not be completed due to errors during file processing.")

print("-" * 30)
print("Script finished.")