"""
Electron supply functions.

This module defines how the electrons arriving at the barrier are distributed
in energy, set by the Fermi level and the temperature: the Fermi-Dirac
occupancy ``f(E)``, which builds the total energy distribution, and the supply
``l(E) = k_B T ln(1 + exp(-(E - E_F)/k_B T))``, the occupancy integrated over
transverse momentum, which builds the normal energy distribution.

Dependencies
------------
numpy : Array manipulation and vectorized element-wise math.
getelec.constants : Domain physical constants library.
abc : Base structure handling for abstract classes.
typing : Type annotation management tools.
"""

import numpy as np
from scipy.special import expit
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

    def get_occupancy(self, energy_array: np.ndarray) -> np.ndarray:
        """
        Fermi-Dirac occupancy f(E), dimensionless and in [0, 1].

        Distinct from :meth:`get_supply`, and the distinction matters. The
        supply function is the log term

            l(E) = k_B T ln(1 + exp(-(E - E_F) / k_B T)) ,

        which is the occupancy already integrated over transverse momentum. It
        is what multiplies D(E) in the current integral, and therefore what the
        **normal** energy distribution is built from. The **total** energy
        distribution instead pairs the bare occupancy with the transmission
        integrated over normal energy,

            NED(E) = l(E) D(E) ,      TED(E) = f(E) * integral of D dE_z .

        The two are related by ``dl/dE = -f``, which is exactly why both
        integrate to the same current density -- integrating one by parts gives
        the other. Using the wrong one gives a curve that looks plausible,
        peaks in nearly the right place, and integrates to the wrong number.

        Returns
        -------
        np.ndarray
        """
        energy_array = np.asarray(energy_array, dtype=float)
        if self.temperature < 1e-6:
            return np.where(energy_array < self.fermi_level, 1.0, 0.0)
        return expit(-(energy_array - self.fermi_level)
                     / (const.KB * self.temperature))


    def get_log_supply(self, energy_array: np.ndarray) -> np.ndarray:
        """
        Supply function l(E) = k_B T ln(1 + exp(-(E - E_F) / k_B T)), in eV.

        The occupancy already integrated over transverse momentum. This is what
        multiplies the transmission in the current integral and in the normal
        energy distribution, while :meth:`get_occupancy` gives ``f(E)`` for the
        total energy distribution. They satisfy ``dl/dE = -f``.

        Provided on the base class so that neither distribution depends on which
        supply object happens to be attached to the emitter: a ``FermiDirac``
        supply returns ``f`` from ``get_supply`` and a ``LogFermiDirac`` returns
        ``l``, but both are determined by the Fermi level and the temperature,
        so both are always available.

        Returns
        -------
        np.ndarray
        """
        energy_array = np.asarray(energy_array, dtype=float)
        if self.temperature < 1e-6:
            return np.maximum(self.fermi_level - energy_array, 0.0)
        thermal = const.KB * self.temperature
        return thermal * np.logaddexp(0.0, -(energy_array - self.fermi_level) / thermal)

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

        # f(E) = 1 / (1 + exp(x)) is exactly the logistic function of -x.
        # scipy's expit is stable over the whole real line, so no branching and
        # no overflow -- the previous np.where evaluated both branches and
        # produced inf/inf warnings for large |x| before discarding one of them.
        distribution = expit(-x)

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
        r"""
        Calculate the log-form integrated electronic supply spectrum or its density-weighted equivalent.

        ``k_B T ln(1 + exp(-(E - E_F)/k_B T))``, in eV, evaluated with
        ``numpy.logaddexp``, which is exact in both tails; at T = 0 it is
        ``max(E_F - E, 0)``.

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
            # T -> 0 limit of k_B T ln(1 + exp(-(E - E_F)/k_B T)), which is
            # (E_F - E) below the Fermi level and zero above it. No k_B T
            # survives, which is exactly why the caller must not multiply this
            # by the temperature -- see the note in get_supply's docstring.
            return np.where(energy_array < self.fermi_level,
                            self.fermi_level - energy_array, 0.0)
        else:

            x = -(energy_array - self.fermi_level) / (const.KB * self.temperature)

            # log(1 + exp(x)) evaluated stably. np.logaddexp(0, x) is exact in
            # both tails, so the hand-tuned x > 35 / x < -36 cutoffs and the
            # suppressed overflow warning are no longer needed.
            # k_B T ln(1 + exp(-(E - E_F)/k_B T)). The k_B T is included here
            # rather than left to the caller so that this function returns the
            # same physical quantity at every temperature, including zero --
            # where the factor vanishes and no rescaling by T could recover it.
            log_term = const.KB * self.temperature * np.logaddexp(0.0, x)

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
            