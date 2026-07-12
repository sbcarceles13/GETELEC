import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os
if '__file__' in locals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
from getelec.potential_barrier import SchottkyPotential
from getelec.transmission_solver import Noumerov
from getelec.electron_supply import FermiDirac
from getelec.band_structure import CustomMetal
from getelec.electron_emitter import MetalEmitter
watermark_text = "by GETELEC"

def main():
    print("Loading TED Data...")
    # 1. Load the data
    path = r".\examples\ted.txt"
    data = np.loadtxt(path)
    energy = data[:,0]
    energy_exp = energy + 90
    counts = data[:,1]

    print("Setting up Fitting Model...")
    my_potential = SchottkyPotential()
    my_band = CustomMetal(energy_exp)
    my_solver = Noumerov()
    my_supply = FermiDirac()

    emitter = MetalEmitter(my_potential, my_solver, my_supply, my_band)

    initial_guess = [4, 90, 4.5, 300]   # field, fermi_level, work_function, temperature
    bounds = ([1, 89.9, 3, 100], [10, 90.1, 6, 2000])
    norm_counts = counts/max(counts)

    def model(x_energies, f,fl, wf, temp):
 
        emitter.update_params(field=f, work_function=wf, fermi=fl, temp=temp)
        
        _, ted = emitter.calculate_total_energy_distribution()

        return ted

    print("Fitting Data...")
    popt, pcov = curve_fit(model, energy_exp, norm_counts, p0=initial_guess, bounds=bounds,maxfev=1000000)

    print(f"    Fitted param: F = {popt[0]:.2f} V/nm, E_F = {popt[1]:.2f} eV, PHI = {popt[2]:.2f} eV, T = {popt[3]:.2f} K")

    emitter.update_params(field=popt[0], work_function=popt[2], fermi=popt[1], temp=popt[3])
    _, fitted = emitter.calculate_total_energy_distribution()
    
    # 4. Plot the results
    print("Plotting distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left Plot: Linear Scale
    ax = axes[0]
    ax.scatter(energy,norm_counts,c='black', s=5,label="Data points")
    ax.plot(energy,fitted,label="Numerical fit")
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
    ax.scatter(energy,norm_counts,c='black', s=5,label="Data points")
    ax.plot(energy,fitted,label="Numerical fit")
    ax.axvline(x=0, color='r', linestyle='--', label='Fermi Level ($E_F$)')
    ax.set_xlabel("Energy relative to Fermi Level (eV)")
    ax.set_ylabel("Normalized Distribution (a.u.)")
    ax.set_title("Field Emission Energy Distributions (Log Scale)")
    ax.legend()
    ax.set_yscale('log') # Set Y-axis to Log Scale
    ax.set_ylim(bottom=1e-6)
    ax.grid(True, alpha=0.3, which='both')
    ax.text(0.99, 0.01, watermark_text, transform=ax.transAxes,
            fontsize=10, color='gray', alpha=0.5,ha='right',
            va='bottom', rotation=0)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
