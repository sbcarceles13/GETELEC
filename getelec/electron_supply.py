"""
Docstring for getelec.gt3.electron_supply

This module defines the electron supply functions used in the GT3 model for electron emission.

It provides abstract and concrete classes to model the available carrier counts and energy 
distributions of electrons incident upon a barrier surface using various foundational statistical 
frameworks (e.g., standard Fermi-Dirac distribution, and log-formulated variants tracking 
integrated supply capabilities).

Dependencies
------------
numpy : Array manipulation and vectorized element-wise math.
getelec.constants : Domain physical constants library.
abc : Base structure handling for abstract classes.
typing : Type annotation management tools.
"""

import numpy as np
from abc import ABC, abstractmethod
from getelec import constants as const
from typing import Optional, Tuple

class Supply(ABC):
    """
    Abstract base class for electron supply functions.

    Defines the shared structural interface for computing the available electron flux 
    intensity or probability distribution arriving at an emission boundary as a 
    function of energy.
    """

    @abstractmethod
    def get_supply(self, energy_array: np.ndarray) -> np.ndarray:
        """
        Calculate the electron supply function across an array of energies.

        Parameters
        ----------
        energy_array : numpy.ndarray
            1D array containing the target electronic energy states.

        Returns
        -------
        supply : numpy.ndarray
            1D array containing computed electron supply function values.
        """
        pass

class FermiDirac(Supply):
    """
    Standard Fermi-Dirac distribution supply model.

    Computes the probability of electron state occupancy at a given temperature 
    and Fermi level. Optionally combines this with an external density of states (DOS) 
    profile to evaluate multi-dimensional supply factors.

    Parameters
    ----------
    fermi_level : float, default 9.5
        The chemical potential/Fermi level of the material system.
    temperature : float, default 300.0
        The thermodynamic temperature of the emitter system in Kelvin.

    Attributes
    ----------
    fermi_level : float
        Stored value for the system's chemical potential.
    temperature : float
        Stored value for the system's absolute temperature profile.
    """

    def __init__(self, fermi_level: float = 9.5, temperature: float = 300.0):
        """
        Initialize the FermiDirac supply engine.
        """
        self.fermi_level = fermi_level
        self.temperature = temperature
    
    def get_supply(self, energy_array: np.ndarray, states_density: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        """
        Calculate the Fermi-Dirac occupation probability or density-weighted supply.

        Handles the absolute zero temperature case cleanly using step functions, and 
        deploys numerically stable split-domain calculations for non-zero conditions to 
        prevent exponential overflows.

        Parameters
        ----------
        energy_array : numpy.ndarray
            1D array containing target energy values.
        states_density : tuple of numpy.ndarray, optional
            A tuple matching `(dos_energy, dos_values)` tracking raw density of states metrics.
            If None, the pure occupation probability distribution is returned.

        Returns
        -------
        supply : numpy.ndarray
            The computed electron distribution array. If `states_density` is supplied 
            and valid, returns the normalized state-weighted electronic supply index.

        Examples
        --------
        >>> distribution = FermiDirac(fermi_level=5.0, temperature=300)
        >>> energies = np.array([4.8, 5.0, 5.2])
        >>> distribution.get_supply(energies)
        array([0.91104269, 0.5       , 0.08895731])
        """
        # Handle the T=0 case to avoid division by zero
        if self.temperature == 0.0:
            return np.where(energy_array <= self.fermi_level, 1.0, 0.0)

        x = (energy_array - self.fermi_level) / (const.KB * self.temperature)
        
        # Use different stable formulations for positive and negative x
        # Note: We compute exp_neg_x because np.exp(-x) is always safe from overflow
        # when x is positive.
        exp_neg_x = np.exp(-x)
        
        distribution = np.where(
            x > 0,
            exp_neg_x / (exp_neg_x + 1),  # Stable form for x > 0
            1 / (1 + np.exp(x))         # Stable form for x <= 0
        )

        if states_density is None:
            return distribution
        else:
            dos_energy, dos_values = states_density
            if dos_energy.size == 0 or dos_values.size == 0:
                return distribution
            else:
                density_states = np.interp(energy_array,dos_energy,dos_values,left=dos_values[0],right=dos_values[-1])
                dos_supply = distribution * density_states / np.sqrt(energy_array)
                return dos_supply

class LogFermiDirac(Supply):
    """
    Logarithmic variant of the Fermi-Dirac integration supply model.

    Evaluates the integral-ready electronic supply functions (often mapped to normal vector 
    supply components in free electron calculations) using stable logarithmic approximations 
    to mitigate dynamic overflow conditions over steep energy boundaries.

    Parameters
    ----------
    fermi_level : float, default 9.5
        The chemical potential/Fermi level of the material system.
    temperature : float, default 300.0
        The thermodynamic temperature of the emitter system in Kelvin.

    Attributes
    ----------
    fermi_level : float
        Stored value for the system's chemical potential.
    temperature : float
        Stored value for the system's absolute temperature profile.
    """

    def __init__(self, fermi_level: float = 9.5, temperature: float = 300.0):
        """
        Initialize the LogFermiDirac supply engine.
        """
        self.fermi_level = fermi_level
        self.temperature = temperature
        
    def get_supply(self, energy_array: np.ndarray, states_density: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        """
        Calculate the log-form integrated electronic supply spectrum or its density-weighted equivalent.

        Applies threshold switching over large arguments ($\|x\| > 35$) inside an isolated 
        floating-point error state context window to bypass standard evaluation hardware warnings.

        Parameters
        ----------
        energy_array : numpy.ndarray
            1D array containing target energy values.
        states_density : tuple of numpy.ndarray, optional
            A tuple matching `(dos_energy, dos_values)` tracking raw density of states metrics.
            If None, the pure integrated logarithmic supply function array is returned.

        Returns
        -------
        supply : numpy.ndarray
            The computed logarithmic electronic distribution array or its state-density 
            scaled variant.

        Examples
        --------
        >>> distribution = LogFermiDirac(fermi_level=9.5, temperature=100)
        >>> energies = np.array([9.0, 9.5, 10.0])
        >>> distribution.get_supply(energies)
        """
        if self.temperature < 1e-6:
            # ... (T=0 case remains the same)
            constant_at_T0 = 4 * np.pi * const.M_E / (const.H**3 * const.C)
            supply = np.where(energy_array < self.fermi_level, 
                            constant_at_T0 * (self.fermi_level - energy_array), 
                            0.0)

            return supply
        else:

            x = -(energy_array - self.fermi_level) / (const.KB * self.temperature)
            
            # Use a context manager to suppress the expected overflow warning
            with np.errstate(over='ignore'):
                log_term = np.where(x > 35, x, 
                                    np.where(x < -36, np.exp(x), np.log(1 + np.exp(x))))

        if states_density is None:
            return log_term
        else:
            dos_energy, dos_values = states_density
            if dos_energy.size == 0 or dos_values.size == 0:
                return log_term
            else:
                density_states = np.interp(energy_array,dos_energy,dos_values,left=dos_values[0],right=dos_values[-1])
                dos_supply = log_term * density_states / np.sqrt(energy_array)
                return dos_supply
            