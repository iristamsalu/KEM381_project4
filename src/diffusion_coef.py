import numpy as np
import matplotlib.pyplot as plt
import argparse 
import os

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
    if len(time) < 2 or len(msd) < 2 or dimensions not in [2, 3]: # Changed dimensions check
        print("Warning: Not enough data points or invalid dimensions (must be 2 or 3) to compute diffusion coefficient.")
        return np.nan

    # Fit MSD = (2 * d * D) * time + C
    # slope = 2 * d * D
    try:
        # Fit the linear part, start from the one third.
        start_fit_index = len(time) // 3
        # Ensure at least 2 points for the fit
        if start_fit_index >= len(time) - 1:
            # Use all data
            start_fit_index = 0

        fit_time = time[start_fit_index:]
        fit_msd = msd[start_fit_index:]

        if len(fit_time) < 2:
             print("Warning: Not enough data points in the selected range for fitting D. Using all data.")
             fit_time = time
             fit_msd = msd
             # Too little data
             if len(fit_time) < 2:
                  print("Error: Cannot perform linear fit with less than 2 points.")
                  return np.nan

        print(f"Fitting diffusion coefficient using data from time={fit_time[0]:.4f} onwards ({len(fit_time)} points).")
        # Perform linear regression: msd = slope * time + intercept
        slope, intercept = np.polyfit(fit_time, fit_msd, 1)

        if slope < 0:
            print(f"Warning: Calculated slope of MSD vs time is negative ({slope:.2e}). This might indicate issues with simulation or analysis.")
            # Return NaN or 0.0? Returning NaN is safer.
            # return 0.0
            return np.nan

        # D = slope / (2 * d)
        diffusion_coef = slope / (2 * dimensions)
        return diffusion_coef

    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Warning: Linear fit for diffusion coefficient failed: {e}")
        return np.nan

def minimum_image_displacement(displacement, box_size):
    """Apply minimum image convention to calculate displacement component-wise."""
    # Ensure box_size is a numpy array for broadcasting if needed
    box_size_arr = np.asarray(box_size)
    # Avoid division by zero if box size is zero in any dimension
    inv_box_size = np.divide(1.0, box_size_arr, out=np.zeros_like(box_size_arr, dtype=float), where=box_size_arr!=0)
    return displacement - box_size_arr * np.round(displacement * inv_box_size)


def process_trajectory_for_msd(filename, dt_sim, dt_comp, box_size_val, dimensions, skip_interval=1, start_frame=0):
    """
      Reads an XYZ file frame by frame, calculates MSD relative to the specified start frame.

      Args:
          filename (str): Path to the XYZ file.
          dt_sim (float): Simulation timestep duration between frames in the original file.
          dt_comp (float): Computer timestep duration between frames in the original file.
          box_size_val (float): The size of the simulation box (assumed cubic or square).
          dimensions (int): The dimensionality of the system (2 or 3).
          skip_interval (int): Interval for skipping frames.
          start_frame (int): The frame number (0-based index) from which to start the MSD calculation.

      Returns:
          tuple: (msd_array, time_array_sim, time_array_comp, processed_frame_count)
                 Returns (None, None, None, None) if an error occurs.
      """
    # Initialize variables
    msd_values = []
    times_sim = []
    times_comp = []
    n_particles = 0
    first_frame_found = False
    current_time_sim = 0.0
    current_time_comp = 0.0
    current_msd = 0.0

    # Initialize arrays for unwrapped coordinates
    unwrapped_positions = None
    previous_wrapped_positions = None
    first_positions = None

    # Define box array based on dimensions
    box = np.array([box_size_val] * dimensions)

    try:
        with open(filename, 'r') as f:
            line_num = 0
            frame_count_total = 0
            frame_count_processed = 0 # Counts frames actually used (after skipping and starting)

            print(f"Processing trajectory file: {filename}")
            print(f"Parameters: dt_sim={dt_sim}, dt_comp={dt_comp}, box_size={box_size_val}, dimensions={dimensions}, skip={skip_interval}, start={start_frame}")

            while True:
                # Read number of particles
                line1 = f.readline()
                if not line1:
                    break # End of file
                line_num += 1

                try:
                    current_n_particles = int(line1.strip())
                    if frame_count_total == 0:
                        n_particles = current_n_particles
                        print(f"Detected {n_particles} particles.")
                    elif current_n_particles != n_particles:
                         print(f"Warning: Frame {frame_count_total} has {current_n_particles} particles, expected {n_particles}. Skipping frame.")
                         # Skip the rest of this frame
                         f.readline() # comment line
                         for _ in range(current_n_particles):
                             f.readline()
                         line_num += (1 + current_n_particles)
                         frame_count_total += 1
                         continue # Go to next frame read

                except ValueError:
                    print(f"Error reading particle count at line {line_num}. Line content: '{line1.strip()}'")
                    return None, None, None, None

                # Read comment line
                comment_line = f.readline()
                if not comment_line: # Check for EOF after reading comment
                    break
                line_num += 1

                # Check if this frame should be processed based on skip_interval
                if frame_count_total % skip_interval == 0:
                    # Read positions for this frame
                    current_positions_raw = np.zeros((n_particles, 3)) # Read as 3D first
                    try:
                        for i in range(n_particles):
                            line = f.readline()
                            if not line:
                                raise EOFError(f"Unexpected end of file while reading coordinates for frame {frame_count_total}")
                            parts = line.split()
                            # Try to detect if first column is atom type or coordinate
                            offset = 0
                            try:
                                # If the first part is not a number, assume it's an atom type
                                float(parts[0])
                            except ValueError:
                                offset = 1
                            # Ensure enough parts exist
                            if len(parts) < offset + 3:
                                raise ValueError(f"Insufficient coordinate data on line {line_num + i + 1}")

                            current_positions_raw[i] = [float(parts[offset+j]) for j in range(3)]
                        line_num += n_particles
                    except (ValueError, IndexError, EOFError) as e:
                         print(f"Error reading coordinates at frame {frame_count_total}, around line {line_num}: {e}")
                         return None, None, None, None
                    except Exception as e: # Catch other potential errors
                         print(f"An unexpected error occurred reading frame {frame_count_total}: {e}")
                         return None, None, None, None


                    # Check if this frame is the start frame or later
                    current_processed_frame_index = frame_count_total // skip_interval
                    if current_processed_frame_index >= start_frame:
                        # Initialize on the actual start frame
                        if not first_frame_found:
                            print(f"Starting MSD calculation from frame {frame_count_total} (processed index {current_processed_frame_index}).")
                            # Use only relevant dimensions
                            first_positions = current_positions_raw[:, :dimensions].copy()
                            unwrapped_positions = current_positions_raw[:, :dimensions].copy()
                            previous_wrapped_positions = current_positions_raw[:, :dimensions].copy()

                            msd_values.append(0.0)
                            times_sim.append(0.0)
                            times_comp.append(0.0)
                            first_frame_found = True
                            frame_count_processed = 1 # Start counting processed frames from here

                        # Calculate MSD for subsequent frames
                        else:
                            current_wrapped = current_positions_raw[:, :dimensions]

                            # Calculate displacement between consecutive frames
                            frame_displacement = current_wrapped - previous_wrapped_positions
                            # Apply minimum image convention using the correct box dimensions
                            frame_displacement = minimum_image_displacement(frame_displacement, box)

                            # Update unwrapped positions
                            unwrapped_positions += frame_displacement

                            # Calculate MSD using unwrapped coordinates relative to first_positions
                            total_displacement = unwrapped_positions - first_positions
                            squared_displacement = np.sum(total_displacement**2, axis=1)
                            current_msd = np.mean(squared_displacement)

                            # Store results - Time relative to start_frame
                            msd_values.append(current_msd)
                            # Time elapsed since the start_frame was processed
                            current_time_sim = frame_count_processed * dt_sim * skip_interval
                            current_time_comp = frame_count_processed * dt_comp * skip_interval
                            times_sim.append(current_time_sim)
                            times_comp.append(current_time_comp)

                            # Update previous positions for next frame
                            previous_wrapped_positions = current_wrapped.copy()
                            frame_count_processed += 1 # Increment processed frame counter

                            if frame_count_processed % 1000 == 0: # Print progress more often
                                print(f"... Processed frame {frame_count_total} (used frame count: {frame_count_processed}), Sim Time: {current_time_sim:.4f}, MSD: {current_msd:.4e}")

                else:
                    # Skip particle coordinates for non-processed frames efficiently
                    for _ in range(n_particles):
                        if not f.readline(): break # Check for EOF while skipping
                    line_num += n_particles
                    # No need to check for EOF after skipping last coordinate, loop condition handles it

                frame_count_total += 1 # Increment total frame counter


            print(f"\nFinished reading file. Total frames scanned: {frame_count_total}")
            if not first_frame_found:
                 print(f"Warning: Start frame ({start_frame}) was never reached or no frames processed after it.")
                 return np.array([]), np.array([]), np.array([]), 0 # Return empty arrays

            print(f"Total frames used for MSD calculation: {frame_count_processed}")
            return np.array(msd_values), np.array(times_sim), np.array(times_comp), frame_count_processed

    except FileNotFoundError:
        print(f"Error: Input file not found: {filename}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {str(e)}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None, None, None, None


# ==============================================
# --- Main Execution ---
# ==============================================
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Calculate Mean Squared Displacement (MSD) and Diffusion Coefficient from an XYZ trajectory.")

    # Required Arguments
    parser.add_argument("input_file", type=str, help="Path to the input XYZ trajectory file.")
    parser.add_argument("--dt_sim", type=float, required=True, help="Simulation timestep duration between consecutive frames in the XYZ file.")
    parser.add_argument("--dt_comp", type=float, required=True, help="Computer time duration between consecutive frames (e.g., in seconds).")
    parser.add_argument("--box_size", type=float, required=True, help="Size of the simulation box (length for cubic/square).")
    parser.add_argument("--dim", type=int, choices=[2, 3], required=True, help="Dimensionality of the simulation (2 or 3).")

    # Optional Arguments
    parser.add_argument("--skip", type=int, default=1, help="Interval for skipping frames (default: 1).")
    parser.add_argument("--start", type=int, default=0, help="Index (0-based) of the first frame to use for MSD (after skipping, default: 0).")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output plot (default: output).")

    args = parser.parse_args()

    # --- Basic Check ---
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        exit() # Exit if file doesn't exist

    print(f"Starting MSD analysis for '{args.input_file}'...")
    print("-" * 30)

    # Process the trajectory using parsed arguments
    msd, time_sim, time_comp, num_processed_frames = process_trajectory_for_msd(
        args.input_file,
        args.dt_sim,
        args.dt_comp,
        args.box_size,
        args.dim,
        args.skip,
        args.start
    )

    print("-" * 30)

    # Analyze results
    if msd is not None and time_sim is not None and time_comp is not None and num_processed_frames > 0:
        sim_time_processed = time_sim[-1] if len(time_sim) > 0 else 0.0
        comp_time_processed = time_comp[-1] if len(time_comp) > 0 else 0.0

        print("\n--- Analysis Results ---")
        print(f"System Dimensions: {args.dim}D")
        print(f"Number of processed frames used for MSD: {num_processed_frames}")
        print(f"Total simulation time analyzed: {sim_time_processed:.5f} (simulation time units)")
        print(f"Total computer time analyzed: {comp_time_processed:.5f} (s)")

        # Calculate Diffusion Coefficient using the provided dimensions
        D_sim = compute_diffusion_coefficient(msd, time_sim, args.dim)
        D_comp = compute_diffusion_coefficient(msd, time_comp, args.dim)

        if not np.isnan(D_sim) :
            print(f"Estimated Diffusion Coefficient (D) from Simulation Time: {D_sim:.5e}")
        else:
             print("Diffusion Coefficient (from Sim Time) could not be reliably estimated.")

        if not np.isnan(D_comp):
            print(f"Estimated Diffusion Coefficient (D) from Computer Time: {D_comp:.5e}")
        else:
             print("Diffusion Coefficient (from Comp Time) could not be reliably estimated.")

        # Plotting
        try:
            # Create a figure with 2 subplots
            fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=False)

            # --- First Plot: MSD vs. Simulation Time ---
            axes[0].plot(time_sim, msd, '-', linewidth=1.0, color="red", label=f'MSD (Sim. Time)')
            axes[0].set_xlabel(f"Time (Simulation Units, relative to start frame)")
            axes[0].set_ylabel("Mean Squared Displacement (MSD)")
            axes[0].set_title("MSD vs. Simulation Time")
            axes[0].grid(True, linestyle='--', alpha=0.6)
            axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Scientific notation for y-axis

            # Linear Fit for Simulation Time
            if not np.isnan(D_sim) and len(time_sim) > 1:
                start_fit_index = len(time_sim) // 3
                if start_fit_index < len(time_sim) -1: # Ensure enough points
                    fit_time_sim = time_sim[start_fit_index:]
                    fit_msd_sim = msd[start_fit_index:]
                    if len(fit_time_sim) >= 2:
                        slope_sim = 2 * args.dim * D_sim
                        intercept_sim = np.mean(fit_msd_sim) - slope_sim * np.mean(fit_time_sim)
                        axes[0].plot(fit_time_sim, slope_sim * fit_time_sim + intercept_sim, '--', color='black', linewidth=2, label=f'Linear Fit (D={D_sim:.2e})')
                else:
                     print("Not enough points for linear fit plot (Sim Time).")


            axes[0].legend()

            # --- Second Plot: MSD vs. Computer Time ---
            axes[1].plot(time_comp, msd, '-', linewidth=1.0, color="blue", label=f'MSD (Comp. Time)')
            axes[1].set_xlabel(f"Time (seconds, relative to start frame)")
            axes[1].set_ylabel("Mean Squared Displacement (MSD)")
            axes[1].set_title("MSD vs. Computer Time")
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Scientific notation for y-axis


            # Linear Fit for Computer Time
            if not np.isnan(D_comp) and len(time_comp) > 1:
                 start_fit_index = len(time_comp) // 3
                 if start_fit_index < len(time_comp) -1: # Ensure enough points
                    fit_time_comp = time_comp[start_fit_index:]
                    fit_msd_comp = msd[start_fit_index:]
                    if len(fit_time_comp) >= 2:
                        slope_comp = 2 * args.dim * D_comp
                        intercept_comp = np.mean(fit_msd_comp) - slope_comp * np.mean(fit_time_comp)
                        axes[1].plot(fit_time_comp, slope_comp * fit_time_comp + intercept_comp, '--', color='black', linewidth=2, label=f'Linear Fit (D={D_comp:.2e})')
                 else:
                     print("Not enough points for linear fit plot (Comp Time).")


            axes[1].legend()

            # Adjust layout and save to output directory
            plt.tight_layout()

            # --- Ensure output directory exists ---
            output_directory = args.output_dir
            os.makedirs(output_directory, exist_ok=True)

            # Create filename and save
            base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
            plot_filename = os.path.join(output_directory, f"msd_plot_{base_filename}.png")
            plt.savefig(plot_filename)
            print(f"\nMSD plot saved to {plot_filename}")

        except ImportError:
            print("\nMatplotlib not found. Cannot generate plot.")
        except Exception as e:
            print(f"\nAn error occurred during plotting: {e}")

    elif num_processed_frames == 0 and msd is not None:
         print("\nNo frames were processed based on the specified start frame and skip interval.")
    else:
        print("\nAnalysis could not be completed due to errors during file processing.")

    print("-" * 30)
    print("Script finished.")