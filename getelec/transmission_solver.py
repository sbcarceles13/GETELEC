"""
Numerical solvers for calculating quantum transmission probabilities.

Provides the `TransmissionSolver` interface and specific integration 
methods, such as the Numerov algorithm, to solve the 1D Schrödinger equation 
across a given potential barrier.
"""

import os
import re
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from numba import jit, prange
from scipy.integrate import quad
import joblib
from pathlib import Path
from getelec import constants as const
from getelec import potential_barrier

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _run_noumerov_batch(psi_batch: np.ndarray, T_vals_batch: np.ndarray) -> np.ndarray:
    """
    Internal Numba-optimized batch solver for the Numerov algorithm.
    
    Parameters
    ----------
    psi_batch : np.ndarray
        2D array representing the wavefunctions to be updated in-place.
    T_vals_batch : np.ndarray
        2D array of the pre-calculated T values for the Numerov integration.

    Returns
    -------
    np.ndarray
        The updated wavefunction array.
    """
    num_energies = psi_batch.shape[0]
    N = psi_batch.shape[1]
    for i in prange(num_energies):
        for j in range(1, N - 1):
            numerator = (2 + 10 * T_vals_batch[i, j]) * psi_batch[i, j] - (1 - T_vals_batch[i, j - 1]) * psi_batch[i, j - 1]
            denominator = 1 - T_vals_batch[i, j + 1]
            psi_batch[i, j + 1] = numerator / denominator
    return psi_batch

class TransmissionSolver(ABC):
    """
    Abstract base class for all transmission solvers.
    
    Defines the mandatory interface for calculating quantum transmission 
    probabilities across a spatial potential barrier.
    """
    @abstractmethod
    def calculate_transmission(self, potential: potential_barrier, energies: np.ndarray) -> np.ndarray:
        """
        Calculates the transmission probability for given energies.

        Parameters
        ----------
        potential : potential_barrier.Barrier
            The potential barrier object containing the spatial voltage profile.
        energies : np.ndarray
            1D array of energy levels to calculate transmission for (in eV).

        Returns
        -------
        np.ndarray
            1D array of transmission probabilities [0.0 to 1.0].
        """
        pass

class Noumerov(TransmissionSolver):
    """
    Numerov method solver for the 1D Schrödinger equation.

    Uses a highly stable 4th-order numerical integration method to calculate
    transmission probabilities across an arbitrary potential barrier.

    Parameters
    ----------
    x_metal : int or float, default -1
        Starting coordinate inside the metal (nm).
    x_vac_plus : int or float, default 10
        Distance into the vacuum to continue integration (nm).
    h : float, default 1e-3
        Spatial step size for the integration grid.
    max_barrier_width : float, default 3.0
        Maximum expected width of the potential barrier (nm).
    """
    def __init__(self, x_metal: int = -1, x_vac_plus: int = 10, h: float = 1e-3, max_barrier_width: float = 3.0):
        # These were arguments in your function, now they are settings.
        self.x_metal = x_metal
        self.x_vac_plus = x_vac_plus
        self.h = h
        self.max_barrier_width = max_barrier_width

    def calculate_transmission(self, potential: potential_barrier, energies: np.ndarray) -> np.ndarray:
        """
        Executes the Numerov integration for the provided potential.

        Parameters
        ----------
        potential : potential_barrier.Barrier
            The specific barrier to integrate across.
        energies : np.ndarray
            1D array of energy levels.

        Returns
        -------
        np.ndarray
            Calculated transmission probabilities for each energy.
        """
        
        # 1. Setup Spatial Grid (Local to the solver)
        x_start = self.x_metal
        x_end = self.max_barrier_width + self.x_vac_plus
        # Integrates from right-to-left
        x_points = np.arange(x_end, x_start - self.h, -self.h) 
        
        # 2. Get Potential (Calling the Class!)
        V = potential.get_potential(x_points)

        #N = x_points.shape[0]
        M = energies.shape[0]

        # **MODIFICATION**: Removed k_metal, matrix, and matrix_inv.
        # We will compute them per-energy later.
        x_a, x_b = x_points[-1], x_points[-2] # Points inside the metal

        # --- 2. Vectorized Setup for all energies ---
        E_minus_V_batch = (energies[:, np.newaxis] - V[np.newaxis, :]) / const.HBAR2_2M
        T_vals_batch = -(self.h**2 / 12) * E_minus_V_batch
        k_batch = np.sqrt(E_minus_V_batch.astype(np.complex128))
        
        psi_batch = np.zeros_like(k_batch, dtype=np.complex128)
        
        k_vac_batch = k_batch[:, 0] # Wavevector at x_end (in vacuum)
        psi_batch[:, 0] = np.exp(1j * k_vac_batch * x_points[0])
        psi_batch[:, 1] = np.exp(1j * k_vac_batch * x_points[1])
        
        # --- 3. Run the parallelized Numba integration ---
        psi_batch = _run_noumerov_batch(psi_batch, T_vals_batch)
        
        # --- 4. Vectorized final calculation (Corrected) ---
        
        # Get the wavevectors inside the metal (at x_a and x_b) FOR EACH energy
        # k_batch[:, -1] corresponds to k at x_points[-1] = x_a
        # k_batch[:, -2] corresponds to k at x_points[-2] = x_b
        k_a_batch = k_batch[:, -1]
        k_b_batch = k_batch[:, -2]
        
        # Get the calculated wavefunctions at the final two points
        # vectors shape is (M, 2)
        vectors = psi_batch[:, [-1, -2]] 
        
        # Build the stack of 2x2 matrices, shape (M, 2, 2)
        matrix_batch = np.zeros((M, 2, 2), dtype=np.complex128)
        
        matrix_batch[:, 0, 0] = np.exp(1j * k_a_batch * x_a)
        matrix_batch[:, 0, 1] = np.exp(-1j * k_a_batch * x_a)
        matrix_batch[:, 1, 0] = np.exp(1j * k_b_batch * x_b)
        matrix_batch[:, 1, 1] = np.exp(-1j * k_b_batch * x_b)
        
        # Invert the stack of matrices
        matrix_inv_batch = np.linalg.inv(matrix_batch)
        
        # Solve for coefficients [A, B] for all energies
        # We are solving M * C = V  =>  C = M_inv * V
        # Use einsum for batched matrix-vector multiplication:
        # 'mij,mj->mi' means: for M matrices (m), multiply (2,2) (ij) by (2,) (j) -> (2,) (i)
        coeffs = np.einsum('...ij,...j->...i', matrix_inv_batch, vectors)
        
        A_coeffs = coeffs[:, 0] # Incident wave amplitude (1/T in some conventions)
        
        # **MODIFICATION**: Use the energy-dependent k_metal_batch
        # k_metal_batch is the wavevector at the matching point x_a
        k_metal_batch = k_a_batch
        
        # k_vac_batch was already calculated in step 2
        
        # Calculate transmission T = (v_trans / v_inc) * (1 / |A|^2)
        # v = p/m = (hbar*k)/m => v_trans/v_inc = k_vac / k_metal
        transmission_coeffs = (np.real(k_vac_batch) / np.real(k_metal_batch)) * (1 / np.abs(A_coeffs)**2)
        
        # Handle potential evanescent waves (where k is imaginary)
        transmission_coeffs[np.isnan(transmission_coeffs)] = 0.0
    
        return transmission_coeffs

class WBK(TransmissionSolver):
    def __init__(self, fermi_level: float = 9.5, work_function: float = 4.5, electric_field: float = 3.0):
        self.fermi_level = fermi_level
        self.work_function = work_function
        self.electric_field = electric_field

    def _get_potential_barrier_height(self, x: float) -> float:
        if x <= 0:
            return np.inf
        else:
            return self.fermi_level + self.work_function - self.electric_field * x - const.COULOMB_CONST / (4 * x)

    def _integrand_wkb(self, x: float, E: float) -> float:
        potential_val = self._get_potential_barrier_height(x) - E
        if potential_val <= 0:
            return 0.0
        else:
            return np.sqrt(potential_val)

    def calculate_transmission(self, potential: potential_barrier, energies: np.ndarray) -> np.ndarray:
        # This is the body of 'transmission_wkb'
        transmission_coeffs = np.zeros_like(energies)

        for i, E in enumerate(energies):

            a_quad = self.electric_field
            b_quad = -self.fermi_level - self.work_function + E
            c_quad = -const.COULOMB_CONST / 4

            discriminant = b_quad**2 - 4 * a_quad * c_quad

            if discriminant < 0:
                transmission_coeffs[i] = 0.0
            else:
                sqrt_discriminant = np.sqrt(discriminant)
                x1 = (-b_quad - sqrt_discriminant) / (2 * a_quad)
                x2 = (-b_quad + sqrt_discriminant) / (2 * a_quad)

                integral_value, _ = quad(self._integrand_wkb, x1, x2, args=(E,), limit=100)
                gamow_factor = const.WKB_PREFACTOR * integral_value
                transmission_coeffs[i] = 1/ (1 + np.exp(-2 * gamow_factor))
        return transmission_coeffs

class ML(TransmissionSolver):
    def __init__(self, fermi_level: float = 9.5, work_function: float = 4.5, electric_field: float = 3.0):
        self.fermi_level = fermi_level
        self.work_function = work_function
        self.electric_field = electric_field
        
        _current_dir = Path(__file__).resolve().parent

        self._path_to_models = _current_dir.parent / "machine_memory"
        self._current_model = None
        self._loaded_ef = None 

    def _load_model_if_needed(self):
        ef_rounded = round(self.fermi_level * 2) / 2
        
        if self._current_model is None or self._loaded_ef != ef_rounded:
            model_filename = f"ef_{ef_rounded}.joblib"

            model_path = self._path_to_models / model_filename
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"No model found for ef={ef_rounded} at {model_path}\n"
                    f"Check if this folder exists: {self._path_to_models.absolute()}"
                )
            
            self._current_model = joblib.load(model_path)
            if hasattr(self._current_model, "feature_names_in_"):
                delattr(self._current_model, "feature_names_in_")
            self._loaded_ef = ef_rounded

    def calculate_transmission(self, potential: potential_barrier, energies: np.ndarray) -> np.ndarray:

        self._load_model_if_needed()

        num_energies = len(energies)
        input_matrix = np.column_stack((
            np.full(num_energies, self.fermi_level), 
            np.full(num_energies, self.work_function),
            np.full(num_energies, self.electric_field),
            energies
        ))

        transmission_coeffs = self._current_model.predict(input_matrix)

        return np.exp(transmission_coeffs)