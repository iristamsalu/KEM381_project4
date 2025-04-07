# KEM381
## Project Assignment 4

### Overview
This is a simple Lennard-Jones MD model in 2D and 3D. It is implemented as a molecular dynamics simulation where you can use different boundary conditions, computational algorithms, Langevin and Berendsen thermostats and change system parameters. Also, computational time, energies and trajectory are tracked during the simulation. Additionally, it has two separate programs that calculate radial distribution function (RDF) and diffusion coefficient. The project offers two main functionalities:

1. It simulates particles under the influence of the **Lennard-Jones Potential** using **Velocity Verlet** algorithm.

2. **Energy Minimization** involves finding the configuration of particles that minimizes the potential energy of the system by optimizing particle positions.


### Files
- **main.py**: Main execution script that parses arguments, initializes the simulation, chooses between algorithms, and runs either energy minimization or full Lennard-Jones simulation.
- **simulation.py**: Core of the simulation logic. Handles initialization, Velocity Verlet integration, energy minimization, boundary conditions, force calculations, lattice setup, and velocity initialization.
Provides simulate_LJ() for full dynamics and minimize_energy() for energy minimization.
- **thermostats.py**: Contains two thermostat functions: Langevin and Berendsen.
- **forces.py**: Implements Lennard-Jones force and potential calculations using both the naive pairwise method and the optimized LCA for efficient neighbor searching.
- **forces_jit.py**: Has same functions as forces.py but uses just-in-time compilation.
- **config.py**: Parses and validates command-line arguments, and stores simulation parameters in a structured Configuration dataclass.
- **output_and_plots.py**: Handles visualization and trajectory saving.
- **RDF.py**: Computes and plots RDF
- **diffusion_coeffcient.py**: Computes diffusion coeffcient from positions MSD vs. simulation time and MSD vs. computer time plots.
- **requirements.txt**: A list of packages and libraries needed to run the programs.


### Installing Requirements
To install the necessary requirements in a virtual environment, use the following command:
pip3 install -r requirements.txt

### Running the Main MD Program
To run the MD simulation program, you need to provide certain parameters through the command line.

#### Explanation of Arguments:

- `--dimensions <2 or 3>`: Set simulation to 2D or 3D (default: 2)
- `--steps <number_of_steps>`: The number of steps to run the simulation (default: 5000)
- `--dt <time_step>`: The time step used in the simulation (default: 0.0001)
- `--density <density>`: The particle density in the system (default: 0.8)
- `--n_particles <number_of_particles>`: The number of particles in the simulation (default: 100)
- `--use_pbc`: (Optional) Flag to enable **Periodic Boundary Conditions**. If omitted, **hard wall** boundary conditions are used by default (default: False)
- `--sigma <LJ_sigma>`: (Optional) The Lennard-Jones sigma parameter (distance where the potential is zero) (default: 1)
- `--epsilon <LJ_epsilon>`: (Optional) The Lennard-Jones epsilon parameter (depth of the potential well) (default: 1)
- `--rcutoff <LJ_cutoff_radius>`: (Optional) The cutoff radius for the Lennard-Jones potential (default: 2.5)
- `--minimize`: (Optional) Flag to run **energy minimization** from random particle positions. If omitted, the regular **Lennard-Jones simulation** will be run by default (default: False)
- `--minimize_only`: (Optional) Flag to run **energy minimization** from initial lattice. If omitted, the regular **Lennard-Jones simulation** will be run by default (default: False)
- `--minimization_steps`: (Optional) Give only when running with `--minimize` or `--minimize_only` (default: 10000)
- `--use_lca`: (Optional) Flag to run the LJ simulation or minimization using the **linked cell algorithm**. If omitted, the regular naive algorithm will be used by default (default: False)
- `--use_jit`: (Optional) Flag to run the LJ simulation or minimization using the **just-in-time compilation (JIT)** (default: False)
- `--temperature <temperature>`: (Optional) The temperature in reduced units (default: 0.5)
- `--use_langevin`: (Optional) Flag to use **Langevin thermostat** (default: False)
- `--use_berendsen`: (Optional) Flag to use **Berendsen thermostat** (default: False)
- `--thermostat_constant`: (Optional) Specify thermostat parameter, for Langevin zeeta or for Berendsen tau. (default: 1)

#### Example Commands:
- **Example 1: NVE With Periodic Boundary Conditions (PBC)**:
    ```
    python3 main.py --dimensions <2 or 3> --steps <number_of_steps> --dt <time_step> --density <density> --n_particles <number_of_particles>

    python3 main.py --dimensions 2 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20 --use_pbc --use_lca --use_jit 

    ```

- **Example 2: Minimize energy starting from initial lattice**:
    ```
    python3 main.py --dimensions <2 or 3> --dt <time_step> --density <density> --n_particles <number_of_particles> --minimize_only --minimization_steps <number_of_steps>

    python3 main.py --dimensions 2 --dt 0.0001 --density 0.8 --n_particles 20 --minimize_only --minimization_steps 10000 --use_lca --use_jit

    ```

- **Example 3: Minimize energy running NVE simulation with PBC before minimization**:
    ```
    python3 main.py --dimensions <2 or 3> --steps <number_of_steps_before_minimization> --dt <time_step> --density <density> --n_particles <number_of_particles> --use_pbc --minimize --minimization_steps <number_of_minimization_steps>

    python3 main.py --dimensions 3 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 100 --minimize --minimization_steps 10000 --use_lca --use_jit

    ```

- **Example 4: Run MD simulation with Langevin thermostat and PBC**:
    ```
    python3 main.py --dimensions <2 or 3> --steps <number_of_steps_before_minimization> --dt <time_step> --density <density> --n_particles <number_of_particles> --use_pbc --use_langevin --temperature <desired temperature> --thermostat_constant <langevin zeeta> --use_lca --use_jit

    python3 main.py --dimensions 3 --steps 10000 --dt 0.01 --density 0.8 --n_particles 100 --use_pbc --use_langevin --temperature 0.5 --thermostat_constant 1 --use_lca --use_jit

    ```

- **Example 5: Run MD simulation with Berendsen thermostat and PBC**:
    ```
    python3 main.py --dimensions <2 or 3> --steps <number_of_steps_before_minimization> --dt <time_step> --density <density> --n_particles <number_of_particles> --use_pbc --use_berendsen --temperature <desired temperature> --thermostat_constant <berendsen tau> --use_lca --use_jit

    python3 main.py --dimensions 3 --steps 10000 --dt 0.01 --density 0.8 --n_particles 100 --use_pbc --use_berendsen --temperature 0.7 --thermostat_constant 1 --use_lca --use_jit

    ```

#### MD simulation output:
1. Program creates output folder and several output files.
2. Trajectory is saved to an `.xyz` file which can be visualized using tools like VMD or Ovito.
3. The energy evaluation plot is saved as `energy_plot.png`. Detailed energy values during the simulation are stored in `energy_data.dat`.
4. Run history including run parameters and computational times is saved to `run_history.dat`.

---

### Running RDF.py
To calculate RDF you need to have a .XYZ file of a trajectory.

#### Explanation of Arguments:

- `<filename>`: (required) XYZ file that you want to use or path to the input XYZ file
- `--density <density>`: (required) The particle density in the system
- `--dim <2 or 3>`: (required) Is the XYZ in 2D or 3D
- `--start <starting frame for RDF calculations>`: (optional) Give a starting frame if you want to skip some of the inital frames (default: 0)
- `--bins <nr of histogram bins>`: (optional) Number of RDF histogram bins for resolution (default: 100)

#### Example Commands:
- **Example 1: Compute RDF for a simulation trajectory in ./output/2D_trajectory.xyz**:
    ```
    python3 RDF.py <path to the XYZ file> --density <density> --dim <2D or 3D> --start <starting frame nr> --bins <nr of histogram bins>

    python3 RDF.py ./output/2D_trajectory.xyz --density 0.5 --dim 2 --start 5000 --bins 50

    ```


#### RDF.py output:
1. Program creates a rdf_plot.png file in the output folder.

---

### Running diffusion_coef.py
To compute diffusion coeffcient by plotting MSD vs. simulation time and MSD vs. computer time.

#### Explanation of Arguments:

- `<filename>`: (required) path to the input XYZ file
- `--dt_sim <simulation timestep length>`: (required) Simulation timestep length
- `--dt_comp <timestep length in seconds>`: (required) Computer timestep length in seconds, you can find it from run_history.dat
- `--box_size <length of a box>`: (required) Give box side length, in 2D (nr_of_particles / density) ** (1 / 2), in 3D (nr_of_particles / density) ** (1 / 3)
- `--dim <2 or 3>`: (required) Is the XYZ in 2D or 3D
- `--start <starting frame>`: (optional) Give a starting frame if you want to skip some of the inital frames (default: 0)
- `--skip <skip interval>`: (optional) Use it to skip some frames (default: 1), which means all the frames are used

#### Example Commands:
- **Example 1: Compute diffusion coefficient for a simulation trajectory in ./output/2D_trajectory.xyz**:
    ```
    python3 diffusion_coef.py <path to the XYZ file> --density <density> --dim <2D or 3D> --start <starting frame nr> --bins <nr of histogram bins>

    python3 diffusion_coef.py ./output/2D_trajectory.xyz --dt_sim 0.01 --dt_comp 0.0256 --box_size 20 --dim 2

    ```

#### Output from diffusion_coef.py:
1. Program creates a msd_plot_{filename}.png file in the output folder.


---

### Notes:
The `videos/` folder contains OVITO visuals (.mp4 files) for different phases: gas.mp4, liquid.mp4, hexatic.mp4 and solid.mp4.

---

