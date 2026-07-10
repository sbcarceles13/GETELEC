"""
Field Emitter Execution and Plotting Interface.

This script orchestrates a field emission execution run by setting up simulation parameters, 
instantiating physical components (potential, band structure, transmission solver, and electron 
supplies), calculating both Normal and Total Energy Distributions (NED/TED), and rendering 
comparative linear and logarithmic visualization plots via `matplotlib`.

Attributes
----------
watermark_text : str
    Global identification annotation appended to generated figure viewports.
"""

import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from getelec.potential_barrier import SchottkyPotential
from getelec.band_structure import SmartMetal
from getelec.transmission_solver import Noumerov
from getelec.electron_supply import FermiDirac, LogFermiDirac
from getelec.electron_emitter import MetalEmitter

watermark_text = "by GETELEC"

def main():
    """
    Execute the field emitter simulation pipeline and render diagnostic plots.

    Performs the following linear sequence of tasks:
    1. Instantiates configuration parameters for Fermi level, work function, field, and temperature.
    2. Constructs concrete component implementations (`SchottkyPotential`, `SmartMetal`, `Noumerov`, 
       `LogFermiDirac`, and `FermiDirac`).
    3. Bundles objects inside a `MetalEmitter` manager to extract emission distributions.
    4. Computes normal energy distributions (NED) and total energy distributions (TED).
    5. Formats a dual-axes matplotlib plot mapping distribution vectors over linear and log dimensions.

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
    ELECTRIC_FIELD = 3.0
    TEMPERATURE = 300.0

    # 2. Setup emitter components
    my_potential = SchottkyPotential(fermi_level=FERMI_LEVEL, work_function=WORK_FUNCTION, electric_field=ELECTRIC_FIELD)
    my_band = SmartMetal(barrier_width=3.0, supply_threshold=1e-14, energy_resolution=0.01)
    my_solver = Noumerov(x_metal=-1.0, x_vac_plus=10, h=0.001, max_barrier_width=3)
    my_supply_log = LogFermiDirac(fermi_level=FERMI_LEVEL, temperature=TEMPERATURE)
    my_supply = FermiDirac(fermi_level=FERMI_LEVEL, temperature=TEMPERATURE)

    print("Running simulations...")
    # 3.1. Build emitter for NED
    emitter = MetalEmitter(my_potential, my_solver, my_supply_log, my_band)
    energies, ned = emitter.calculate_normal_energy_distribution()

    # 3.2. Build emitter for TED
    emitter.supply = my_supply
    _, ted = emitter.calculate_total_energy_distribution()

    # 4. Plot the results
    print("Plotting distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left Plot: Linear Scale
    ax = axes[0]
    ax.plot(energies - FERMI_LEVEL, ted/max(ted), label="TED (Total Energy Distribution)")
    ax.plot(energies - FERMI_LEVEL, ned/max(ned), label="NED (Normal Energy Distribution)")
    ax.axvline(x=0, color='r', linestyle='--', label='Fermi Level ($E_F$)')
    ax.set_xlabel("Energy relative to Fermi Level (eV)")
    ax.set_ylabel("Normalized Distribution (a.u.)")
    ax.set_title("Field Emission Energy Distributions (Linear)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.01, watermark_text, transform=ax.transAxes,
            fontsize=10, color='gray', alpha=0.5,ha='right',
            va='bottom', rotation=0)

    # Right Plot: Log Scale
    ax = axes[1]
    ax.plot(energies - FERMI_LEVEL, ted/max(ted), label="TED (Total Energy Distribution)")
    ax.plot(energies - FERMI_LEVEL, ned/max(ned), label="NED (Normal Energy Distribution)")
    ax.axvline(x=0, color='r', linestyle='--', label='Fermi Level ($E_F$)')
    ax.set_xlabel("Energy relative to Fermi Level (eV)")
    ax.set_ylabel("Normalized Distribution (a.u.)")
    ax.set_title("Field Emission Energy Distributions (Log Scale)")
    ax.legend()
    ax.set_yscale('log') # Set Y-axis to Log Scale
    #ax.set_ylim(bottom=1e-6)
    ax.grid(True, alpha=0.3, which='both')
    ax.text(0.99, 0.01, watermark_text, transform=ax.transAxes,
            fontsize=10, color='gray', alpha=0.5,ha='right',
            va='bottom', rotation=0)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()