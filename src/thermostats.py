import numpy as np

def langevin_thermostat(velocities, dt, temperature):
    """Apply Langevin thermostat."""
    # Friction coefficient
    friction_coef = 0.01

    random_force = np.sqrt(2 * friction_coef * temperature * dt) / 2 * np.random.normal(0, 1, velocities.shape)
    # random_force = np.sqrt(6 * friction_coef * temperature * dt) * (np.random.uniform(0, 1, velocities.shape) - 0.5)
    dissipative_force = -friction_coef * velocities
    return dissipative_force, random_force