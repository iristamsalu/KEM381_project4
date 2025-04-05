import numpy as np

def langevin_thermostat(velocities, dt, temperature, friction_coef):
    """Apply Langevin thermostat."""
    random_force = np.sqrt(2 * friction_coef * temperature * dt) * np.random.normal(0, 1, velocities.shape)
    # random_force = np.sqrt(6 * friction_coef * temperature * dt) * (np.random.uniform(0, 1, velocities.shape) - 0.5)
    dissipative_force = -friction_coef * velocities * dt
    return velocities + dissipative_force + random_force

def berendsen_thermostat(velocities, dt, target_temperature, tau, dimensions, n_particles):
    """Apply Berendsen thermostat to rescale velocities."""
    # Compute kinetic energy to measure the current temperature
    kinetic_energy = 0.5 * np.sum(velocities ** 2)
    current_temperature = (2 * kinetic_energy) / (dimensions * n_particles)

    lambda_factor = np.sqrt(1 + (dt / tau) * ((target_temperature / current_temperature) - 1))
    return velocities * lambda_factor