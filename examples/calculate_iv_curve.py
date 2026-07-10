"""
Field Emitter Parameter Sweep and Fowler-Nordheim Plotting Interface.

This script executes an automated parameter sweep over a range of applied electric 
fields to evaluate current density responses ($J$-$F$). It instantiates physical profiles, 
computes iterative transport results, and visualizes characteristics using both log-linear 
and Fowler-Nordheim coordinates via `matplotlib`.

Attributes
----------
watermark_text : str
    Global identification annotation appended to generated figure viewports.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from getelec.potential_barrier import SchottkyPotential
from getelec.band_structure import SmartMetal
from getelec.transmission_solver import Noumerov
from getelec.electron_supply import LogFermiDirac
from getelec.electron_emitter import MetalEmitter

watermark_text = "by GETELEC"

def main():
    """
    Execute the electric field sweep simulation loop and render Fowler-Nordheim diagnostic plots.

    Performs the following linear sequence of tasks:
    1. Instantiates baseline system conditions (Fermi level, work function, and temperature).
    2. Builds a discrete array of sampling electric fields and allocates an array for target output values.
    3. Initializes structural configuration dependencies (`SmartMetal` band model and `Noumerov` solver).
    4. Iterates over the defined field values to dynamically build unique `SchottkyPotential` and 
       `LogFermiDirac` supplies, collecting sequential cumulative current density indices via `MetalEmitter`.
    5. Generates a dual-axes figure plotting raw semi-logarithmic $J$-$F$ values alongside linear 
       Fowler-Nordheim representations ($\ln(J/F^2)$ vs. $1/F$).

    Returns
    -------
    None

    Raises
    ------
    ImportError
        If underlying runtime dependencies from the `getelec` workspace or standard graphic 
        render frameworks are missing.
    """
    print("Setting up Field Emitter...")

    # 1. Setup simulation parametres
    FERMI_LEVEL = 7.5
    WORK_FUNCTION = 4.5
    TEMPERATURE = 300.0

    electric_field = np.arange(3,7.5,0.5)
    current_density = np.zeros_like(electric_field)

    # 2. Setup emitter components
    my_band = SmartMetal(barrier_width=3.0, supply_threshold=1e-14, energy_resolution=0.01)
    my_solver = Noumerov(x_metal=-1.0, x_vac_plus=10, h=0.001, max_barrier_width=3)

    print("Running simulations...")
    for i, f in enumerate(electric_field):

        # 3. Build emitter for J
        my_potential = SchottkyPotential(fermi_level=FERMI_LEVEL, work_function=WORK_FUNCTION, electric_field=f)
        my_supply = LogFermiDirac(fermi_level=FERMI_LEVEL, temperature=TEMPERATURE)

        emitter = MetalEmitter(my_potential, my_solver, my_supply, my_band)

        current_density[i] = emitter.calculate_current_density()


    # 4. Plot the results
    print("Plotting distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left Plot: Linear Scale
    ax = axes[0]
    ax.plot(electric_field, current_density, label="J-F")
    ax.set_xlabel("Electric Field - F (V/nm)")
    ax.set_ylabel("Current Density - J (A/$cm^2$)")
    ax.set_title("J-F (Linear)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.01, watermark_text, transform=ax.transAxes,
            fontsize=10, color='gray', alpha=0.5,ha='right',
            va='bottom', rotation=0)

    # Right Plot: FN coordinates
    ax = axes[1]
    ax.plot(1/electric_field, np.log(current_density/electric_field**2), label="J-F")
    ax.set_xlabel("1 / Electric Field $(V/nm)^{-1}$")
    ax.set_ylabel("log(J/$F^2$)")
    ax.set_title("J-F (Fowler-Nordheim Coordinates)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.01, watermark_text, transform=ax.transAxes,
            fontsize=10, color='gray', alpha=0.5,ha='right',
            va='bottom', rotation=0)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()