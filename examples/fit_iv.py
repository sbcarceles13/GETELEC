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
from getelec.electron_supply import LogFermiDirac
from getelec.electron_supply import FermiDirac
from getelec.band_structure import SmartMetal
from getelec.electron_emitter import MetalEmitter
watermark_text = "by GETELEC"

def main():
    print("Loading I-V Data...")
    # 1. Load the data
    path = r"C:\Users\salvador.barranco-ca\GitLab SBCarceles\gt3\examples\iv.txt"
    data = np.loadtxt(path)
    V_data = data[:,0]
    I_data = data[:,1]

    mask = I_data > 0

    V = V_data[mask]
    I = np.log(I_data[mask])

    print("Setting up Fitting Model...")
    my_potential = SchottkyPotential()
    my_band = SmartMetal()
    my_solver = Noumerov()
    my_supply = FermiDirac()

    emitter = MetalEmitter(my_potential, my_solver, my_supply, my_band)

    initial_guess = [0.01, 1E-18, 9.5, 4.5, 300] # gamma, emission_area, fermi_level, work_function, temperature
    bounds = ([0.001, 1E-20, 7, 3.5, 200], [0.01, 1E-10, 13, 5.5, 700])

    def model(V_array, gamma, area, ef, wf, temp):

        results = np.zeros_like(V_array)

        for i, v in enumerate(V_array):
            f = v * gamma

            emitter.update_params(field=f, work_function=wf, fermi=ef, temp=temp)
        
            j = emitter.calculate_current_density()

            results[i] = j * area * 1E9

        results = np.clip(results,1E-100, None)

        return np.log(results)

    print("Fitting Data...")
    popt, pcov = curve_fit(model, V, I, p0=initial_guess, bounds=bounds,maxfev=1000000)

    print(f"    Fitted param: gamma = {popt[0]:.6f} 1/nm, radius = {1/(5*popt[0]):.2f} nm, area = {popt[1]*1E14:.2f} nm2, ef = {popt[2]:.2f} eV, phi = {popt[3]:.2f} eV, T = {popt[4]:.2f} K")

    FERMI_LEVEL = popt[2]
    WORK_FUNCTION = popt[3]
    TEMPERATURE = popt[4]

    electric_field = V * popt[0]
    current_density = np.zeros_like(electric_field)

    my_band = SmartMetal(barrier_width=3.0, supply_threshold=1e-14, energy_resolution=0.01)
    my_solver = Noumerov(x_metal=-1.0, x_vac_plus=10, h=0.001, max_barrier_width=3)

    print("Running simulations...")
    for i, f in enumerate(electric_field):

        my_potential = SchottkyPotential(fermi_level=FERMI_LEVEL, work_function=WORK_FUNCTION, electric_field=f)
        
        #my_supply = LogFermiDirac(fermi_level=FERMI_LEVEL, temperature=TEMPERATURE)
        my_supply = FermiDirac(fermi_level=FERMI_LEVEL, temperature=TEMPERATURE)

        emitter = MetalEmitter(my_potential, my_solver, my_supply, my_band)

        current_density[i] = emitter.calculate_current_density()

    I_fit = current_density * popt[1] * 1E9

    print("Plotting I-V...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(V, I_data, 'ko', label='Data Points', markersize=4)
    ax.plot(V, I_fit, label='Numerical Fit', linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel('Voltage (V)', fontsize=12)
    ax.set_ylabel('Current (nA)', fontsize=12)
    ax.set_title('I-V Curve (Log Scale)', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.text(0.99, 0.01, watermark_text, transform=ax.transAxes,
            fontsize=10, color='gray', alpha=0.5,ha='right',
            va='bottom', rotation=0)

    ax = axes[1]
    ax.plot(1/V, np.log(I_data/V**2), 'ko', label='Data Points', markersize=4)
    ax.plot(1/V, np.log(I_fit/V**2), label='Numerical Fit', linewidth=2)
    ax.set_xlabel('1/Voltage $(V^{-1})$', fontsize=12)
    ax.set_ylabel('log(I/$V^2$)', fontsize=12)
    ax.set_title('I-V Curve (FN coordinates)', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.text(0.99, 0.01, watermark_text, transform=ax.transAxes,
            fontsize=10, color='gray', alpha=0.5,ha='right',
            va='bottom', rotation=0)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
