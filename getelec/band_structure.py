"""
Band Structure Calculation Module.

This module provides a suite of classes to model and generate energy grids and
band structures for various materials including metals, semiconductors, 
and dynamically optimized ('smart') metallic systems.

"""
# Naming Conventions followed in this module:
# - use 'generate' when creating grids, arrays, or meshes.
# - use 'get' when the task is light (can be used 10,000 times without slowdown).
# - use 'calculate' when the task is computationally heavy.

import numpy as np
from abc import ABC, abstractmethod
from getelec import constants as const

class BandStructure(ABC):
    """
    Abstract base class representing a material's energy band structure.

    Parameters
    ----------
    energy_resolution : float, default 0.01
        The step size or resolution for the energy grid discretization, in electron-volts (eV)

    Attributes
    ----------
    energy_resolution : float
        The step size or resolution for the energy grid discretization.
    """
    def __init__(self, energy_resolution: float = 0.01):
        """
        Initialize the base BandStructure instance.
        """
        self.energy_resolution = energy_resolution
        
    @abstractmethod
    def generate_band_structure(self, fermi_level: float, work_function: float, above_barrier: float, electric_field: float, temperature: float) -> np.ndarray:
        """
        Generate the energy grid or band structure for the material.

        Parameters
        ----------
        fermi_level : float
            The Fermi energy level of the system, in electron-volts (eV)
        work_function : float
            The work function of the material, in electron-volts (eV)
        above_barrier : float
            Energy range parameter extending above the barrier height, in electron-volts (eV)
        electric_field : float
            The applied external electric field, in volts per nanometre (V/nm)
        temperature : float
            The thermodynamic temperature of the system, in Kelvin (K)

        Returns
        -------
        numpy.ndarray
            An array representing the computed energy states or grid.
            
        Raises
        ------
        NotImplementedError
            If the subclass does not implement this abstract method, in electron-voltd (eV)

        Examples
        --------
        >>> # Subclasses must implement this method.
        >>> # band_structure_instance.generate_band_structure(9.5, 4.5, 0.0, 3.0, 300.0)
        """
        pass

class Metal(BandStructure):
    """
    Represents a basic uniform metallic band structure grid.

    Parameters
    ----------
    lower_energy_limit : float, default 0.001
        The minimum boundary of the uniform energy grid. It represents the bottom of the conduction band and it is the zero for our energy reference system.
    upper_energy_limit : float, default 30.0
        The maximum boundary of the uniform energy grid. It represents the upper limit of the energy grid and it can be above the top of the potential barrier.
    energy_resolution : float, default 0.01
        The discretization step size for the grid.

    Attributes
    ----------
    lower_energy_lim : float
        The minimum boundary of the uniform energy grid.
    upper_energy_lim : float
        The maximum boundary of the uniform energy grid.
    energy_resolution : float
        The discretization step size for the grid.
    """
    def __init__(self, lower_energy_limit: float = 0.001, upper_energy_limit: float = 30.0, energy_resolution: float = 0.01):
        """
        Initialize the Metal instance with grid boundaries and resolution.
        """
        super().__init__(energy_resolution=energy_resolution)
        
        self.lower_energy_lim = lower_energy_limit
        self.upper_energy_lim = upper_energy_limit
        self.energy_resolution = energy_resolution

    def generate_band_structure(self, **kwargs) -> np.ndarray:
        """
        Generate a uniform energy grid from lower to upper energy limits.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments. Included for compatibility with the 
            base class interface but ignored during calculation.

        Returns
        -------
        numpy.ndarray
            A 1D array of evenly spaced energy values.

        Examples
        --------
        >>> metal = Metal(lower_energy_limit=0.0, upper_energy_limit=2.0, energy_resolution=0.5)
        >>> metal.generate_band_structure()
        array([0. , 0.5, 1. , 1.5, 2. ])
        """
        return np.arange(self.lower_energy_lim, self.upper_energy_lim + self.energy_resolution, self.energy_resolution)

class Semiconductor(BandStructure):
    """
    Represents a semiconductor band structure with conduction and valence bands.

    Parameters
    ----------
    lower_energy_limit : float, default 0.001
        The absolute lowest energy limit for the valence band generation. It represents the bottom of the valence band and it is the zero for our energy reference system.
    top_valence : float, default 12.5
        The energy level corresponding to the top of the valence band.
    fermi_level : float, default 13.0
        The Fermi energy level of the semiconductor system.
    band_gap : float, default 1.12
        The energy band gap separating the valence and conduction bands.
    electron_eff_mass : float, default 1.64
        The effective mass of electrons in the conduction band. It is a float that represents the relative effective mass of the electron when compared with the free electron mass.
    hole_eff_mass : float, default 0.68
        The effective mass of holes in the valence band.  It is a float that represents the relative effective mass of the hole when compared with the free electron mass.
    upper_energy_limit : float, default 30.0
        The absolute maximum energy limit for the conduction band generation.
    energy_resolution : float, default 0.01
        The discretization step size for the base grids.

    Attributes
    ----------
    lower_energy_lim : float
        The absolute lowest energy limit.
    upper_energy_lim : float
        The absolute maximum energy limit.
    energy_resolution : float
        The discretization step size.
    top_valence : float
        The top of the valence band.
    fermi_level : float
        The Fermi energy level.
    band_gap : float
        The semiconductor energy band gap.
    electron_eff_mass : float
        The effective mass of conduction electrons.
    hole_eff_mass : float
        The effective mass of valence holes.
    """
    def __init__(self, lower_energy_limit: float = 0.001, top_valence: float = 12.5, fermi_level: float = 13.0, band_gap: float = 1.12, electron_eff_mass = 1.64, hole_eff_mass = 0.68,  upper_energy_limit: float = 30.0, energy_resolution: float = 0.01):
        """
        Initialize the Semiconductor instance with full band parameters.
        """
        super().__init__(energy_resolution=energy_resolution)
        
        self.lower_energy_lim = lower_energy_limit
        self.upper_energy_lim = upper_energy_limit
        self.energy_resolution = energy_resolution
        self.top_valence = top_valence
        self.fermi_level = fermi_level
        self.band_gap = band_gap
        self.electron_eff_mass = electron_eff_mass
        self.hole_eff_mass = hole_eff_mass

    def generate_band_structure(self, **kwargs) -> np.ndarray:
        """
        Generate unscaled and scaled dispersion grids for conduction and valence bands.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments. Included for compatibility with the 
            base class interface but ignored during calculation.

        Returns
        -------
        tuple of numpy.ndarray
            A 4-element tuple containing:
            - energies_conduction_band1 : Conduction band unscaled energy grid.
            - energies_conduction_band2 : Conduction band scaled by electron effective mass.
            - energies_valence_band1 : Valence band unscaled energy grid.
            - energies_valence_band2 : Valence band scaled by hole effective mass.

        Examples
        --------
        >>> semi = Semiconductor(lower_energy_limit=10.0, top_valence=12.0, band_gap=1.0, upper_energy_limit=14.0, energy_resolution=1.0)
        >>> ec1, ec2, ev1, ev2 = semi.generate_band_structure()
        """
        bottom_conduction = self.top_valence + self.band_gap
        energies_conduction_band1 = np.arange(bottom_conduction, self.upper_energy_lim + self.energy_resolution, self.energy_resolution)
        energies_conduction_band2 = (self.electron_eff_mass * (energies_conduction_band1 - bottom_conduction))  + bottom_conduction
        energies_valence_band1 = np.arange(self.lower_energy_lim, self.top_valence + self.energy_resolution, self.energy_resolution)
        energies_valence_band2 = ((1+self.hole_eff_mass) * energies_valence_band1) - (self.hole_eff_mass * self.top_valence)

        return energies_conduction_band1, energies_conduction_band2, energies_valence_band1, energies_valence_band2
        
class SmartMetal(BandStructure):
    """
    Represents an adaptive metal grid system optimizing energy boundaries dynamically.

    Notes
    -----
    Use only with SN barrier. The rest of the barriers need their own optimisation protocol.
    If not SN barrier, use Metal or Custom.

    Parameters
    ----------
    barrier_width : float, default 3.0
        The spatial width of the potential barrier, beyond which there is effectivively no tunneling.
    supply_threshold : float, default 1e-14
        The numerical threshold constraint used to cap maximum grid energies, beyond which the emitted electrons do not substatially contribute to the total emitted current.
    energy_resolution : float, default 0.01
        The structural resolution step size of the resulting mesh.

    Attributes
    ----------
    barrier_width : float
        The spatial width of the potential barrier.
    supply_threshold : float
        The numerical threshold constraint.
    """
    def __init__(self, barrier_width: float = 3.0, supply_threshold: float = 1e-14, energy_resolution: float = 0.01):
        """
        Initialize the SmartMetal instance with physical boundary parameters.
        """
        super().__init__(energy_resolution)
        self.barrier_width = barrier_width
        self.supply_threshold = supply_threshold

    def _get_min_energy(self, fermi_level: float = 9.5, work_function: float = 4.5, energy_resolution: float = 0.01, electric_field: float = 3.0) -> float:
        """
        Lightweight task to calculate the dynamic lower energy bound floor.

        Parameters
        ----------
        fermi_level : float, default 9.5
            The system Fermi level.
        work_function : float, default 4.5
            The material work function.
        energy_resolution : float, default 0.01
            Grid spacing step used to lock boundaries.
        electric_field : float, default 3.0
            Applied external field influencing barrier suppression.

        Returns
        -------
        float
            The dynamically adjusted minimum energy boundary.
        """
        if electric_field == 0:
            min = fermi_level + work_function

        elif electric_field > 0:
            min = fermi_level + work_function - np.sqrt((electric_field*self.barrier_width)**2+(4*electric_field*const.IMAGE_PREFACTOR))
        else:
            print("Field cannot be lower than 0")
            min = 0.01
        
        min_floor = np.floor(min / energy_resolution) * energy_resolution

        if np.isclose(min, min_floor):
            new_min = min_floor - energy_resolution
        else:
            new_min = min_floor

        if new_min < 0:
            new_min = energy_resolution
            
        return new_min

    def _get_max_energy(self, fermi_level: float = 9.5, energy_resolution: float = 0.01, temperature: float = 300.) -> float:
        """
        Lightweight task to evaluate the dynamic maximum energy ceiling.

        Parameters
        ----------
        fermi_level : float, default 9.5
            The system Fermi level.
        energy_resolution : float, default 0.01
            Grid spacing step used to align boundaries.
        temperature : float, default 300.0
            System temperature used to account for thermal tail distributions.

        Returns
        -------
        float
            The dynamically adjusted maximum energy boundary. Returns np.nan
            if precision errors or invalid criteria are hit.
        """
        if self.supply_threshold <= 0:
            print("Warning: Target supply must be greater than 0.")
            return np.nan
            
        if temperature < 1e-6:
            constant_at_T0 = 4 * np.pi * const.M_E / (const.H**3 * const.C)
            max = fermi_level - (self.supply_threshold / constant_at_T0)

            max_ceil = np.ceil(max / energy_resolution) * energy_resolution

            if np.isclose(max, max_ceil):
                new_max = max_ceil + energy_resolution
            else:
                new_max = max_ceil

            return new_max
    
        else:
            log_argument = np.expm1(self.supply_threshold)
            
            if log_argument <= 0:
                print("Warning: Calculation for N is at or below numerical precision.")
                return np.nan

            kBT = const.KB * temperature

            max = fermi_level - kBT * np.log(log_argument)
            
            max_ceil = np.ceil(max / energy_resolution) * energy_resolution

            if np.isclose(max, max_ceil):
                new_max = max_ceil + energy_resolution
            else:
                new_max = max_ceil

            return new_max

    def generate_band_structure(self, fermi_level: float = 9.5, work_function: float = 4.5, above_barrier: float = 0., electric_field: float = 3.0, temperature: float = 300.) -> np.ndarray:
        """
        Generate an optimized dynamic array grid bounded by the calculated limits.

        Parameters
        ----------
        fermi_level : float, default 9.5
            The structural Fermi energy level.
        work_function : float, default 4.5
            The target material work function.
        above_barrier : float, default 0.0
            Additional parameter offset over the evaluated potential barrier.
        electric_field : float, default 3.0
            External field strength impacting the profile.
        temperature : float, default 300.0
            The thermal setting in Kelvin.

        Returns
        -------
        numpy.ndarray
            An array containing the structured smart mesh layout.

        Examples
        --------
        >>> sm = SmartMetal()
        >>> sm.generate_band_structure(temperature=100.0)
        """
        energy_low = self._get_min_energy(fermi_level, work_function, self.energy_resolution, electric_field)
        energy_high = self._get_max_energy(fermi_level,self.energy_resolution,temperature)
        return np.arange(energy_low, energy_high + self.energy_resolution, self.energy_resolution)

class Test(BandStructure):
    """
    Test class simulating advanced non-linear mesh formatting over optimized fields.

    Parameters
    ----------
    barrier_width : float, default 3.0
        The width of the tracking potential barrier.
    supply_threshold : float, default 1e-14
        The numerical threshold filter constraint.
    energy_resolution : float, default 0.01
        The step resolution requested for dynamic calculation.

    Attributes
    ----------
    barrier_width : float
        The width of the tracking potential barrier.
    supply_threshold : float
        The numerical threshold filter constraint.
    """
    def __init__(self, barrier_width: float = 3.0, supply_threshold: float = 1e-14, energy_resolution: float = 0.01):
        """
        Initialize the Test instance variables.
        """
        super().__init__(energy_resolution)
        self.barrier_width = barrier_width
        self.supply_threshold = supply_threshold

    def _get_min_energy(self, fermi_level: float = 9.5, work_function: float = 4.5, energy_resolution: float = 0.01, electric_field: float = 3.0) -> float:
        """
        Lightweight calculation tracking potential barrier minimums with safe floors.

        Parameters
        ----------
        fermi_level : float, default 9.5
            The base system Fermi level.
        work_function : float, default 4.5
            The base work function.
        energy_resolution : float, default 0.01
            Grid locking interval constraints.
        electric_field : float, default 3.0
            Applied potential field strength.

        Returns
        -------
        float
            Determined floor for energy minimum configurations.
        """
        if electric_field == 0:
            min = fermi_level + work_function

        elif electric_field > 0:
            min = fermi_level + work_function - np.sqrt((electric_field*self.barrier_width)**2+(4*electric_field*const.IMAGE_PREFACTOR))            
        else:
            print("Field cannot be lower than 0")
            min = 0
        
        min_floor = np.floor(min / energy_resolution) * energy_resolution

        if np.isclose(min, min_floor):
            new_min = min_floor - energy_resolution
        else:
            new_min = min_floor

        if new_min < 0:
            new_min = energy_resolution
            
        return new_min

    def _get_max_energy(self, fermi_level: float = 9.5, energy_resolution: float = 0.01, temperature: float = 300.) -> float:
        """
        Lightweight evaluation checking threshold ceilings for the test mesh.

        Parameters
        ----------
        fermi_level : float, default 9.5
            The base system Fermi level.
        energy_resolution : float, default 0.01
            Locking granularity boundaries.
        temperature : float, default 300.0
            Absolute context temperature.

        Returns
        -------
        float
            Evaluated ceiling constraint value.
        """
        if self.supply_threshold <= 0:
            print("Warning: Target supply must be greater than 0.")
            return np.nan
            
        if temperature < 1e-6:
            constant_at_T0 = 4 * np.pi * const.M_E / (const.H**3 * const.C)            
            max = fermi_level - (self.supply_threshold / constant_at_T0)

            max_ceil = np.ceil(max / energy_resolution) * energy_resolution

            if np.isclose(max, max_ceil):
                new_max = max_ceil + energy_resolution
            else:
                new_max = max_ceil

            return new_max
    
        else:
            log_argument = np.expm1(self.supply_threshold)
            
            if log_argument <= 0:
                print("Warning: Calculation for N is at or below numerical precision.")
                return np.nan

            kBT = const.KB * temperature

            max = fermi_level - kBT * np.log(log_argument)
            
            max_ceil = np.ceil(max / energy_resolution) * energy_resolution

            if np.isclose(max, max_ceil):
                new_max = max_ceil + energy_resolution
            else:
                new_max = max_ceil

            return new_max

    def _generate_mesh(self, E_peak: float = 0 , E_min: float = -1, E_max: float = 1, initial_step: float =0.01, growth_rate: float =1.1, exact_bounds: bool =True):
        """
        Generate a non-linear spaced mesh stretching outwards from a center peak.

        Parameters
        ----------
        E_peak : float, default 0
            The center focal point of the dense mesh grid distribution.
        E_min : float, default -1
            The lower outer constraint bounding development.
        E_max : float, default 1
            The upper outer constraint bounding development.
        initial_step : float, default 0.01
            The beginning step resolution offset spacing from E_peak.
        growth_rate : float, default 1.1
            The compounding percentage geometric scalar step adjustment.
        exact_bounds : bool, default True
            Flag asserting if maximum thresholds must explicitly end on bounds.

        Returns
        -------
        list
            A sequenced ordered array tracking variable mesh energy coordinates.
        """
        right_points = []
        current_E = E_peak
        current_step = initial_step
        
        while current_E < E_max:
            current_E += current_step
            if exact_bounds and current_E > E_max:
                right_points.append(E_max)
                break
            right_points.append(current_E)
            current_step *= growth_rate # Increase step size by 10%

        # 2. Generate points to the LEFT (towards E_min)
        left_points = []
        current_E = E_peak
        current_step = initial_step
        
        while current_E > E_min:
            current_E -= current_step
            if exact_bounds and current_E < E_min:
                left_points.append(E_min)
                break
            left_points.append(current_E)
            current_step *= growth_rate # Increase step size by 10%
            
        # Reverse left points so they go from E_min up to E_peak
        left_points.reverse()
        
        # 3. Combine everything: Left tail + Peak + Right tail
        energy_mesh = left_points + [E_peak] + right_points
        
        return energy_mesh

    def generate_band_structure(self, fermi_level: float = 9.5, work_function: float = 4.5, above_barrier: float = 0., electric_field: float = 3.0, temperature: float = 300.) -> np.ndarray:
        """
        Generate a localized non-linear geometric grid around thermal probability peaks.

        Parameters
        ----------
        fermi_level : float, default 9.5
            Fermi reference baseline.
        work_function : float, default 4.5
            Electronic separation threshold constraint.
        above_barrier : float, default 0.0
            Extended barrier scalar offset.
        electric_field : float, default 3.0
            Field configuration criteria parameters.
        temperature : float, default 300.0
            Thermal setting in Kelvin.

        Returns
        -------
        list
            The compiled compound list configuration representing the dynamic mesh.

        Examples
        --------
        >>> t = Test()
        >>> t.generate_band_structure()
        """
        energy_low = self._get_min_energy(fermi_level, work_function, self.energy_resolution, electric_field)
        energy_high = self._get_max_energy(fermi_level,self.energy_resolution,temperature)

        d = (0.0976*electric_field)/(np.sqrt(work_function))

        p = const.KB*temperature/d

        energy_peak = const.KB * np.log(p/(1-p))

        energy_grid = self._generate_mesh(E_peak = energy_peak, E_min = energy_low, E_max= energy_high)

        return energy_grid  

class CustomMetal(BandStructure):
    """
    Represents a static user-provided pre-defined metal band structure system. It can be used to set any arbitrary metallic band structure - the user should be ware of checking for consistency with the other modules

    Parameters
    ----------
    custom_energy_array : numpy.ndarray
        The pre-computed custom structured array representation.

    Attributes
    ----------
    custom_energy_array : numpy.ndarray
        The pre-computed custom structured array representation.
    """
    def __init__(self, custom_energy_array: np.ndarray):
        """
        Initialize the CustomMetal tracking containing custom array arrays.
        """
        self.custom_energy_array = custom_energy_array

    def generate_band_structure(self, fermi_level: float = 9.5, work_function: float = 4.5, above_barrier: float = 10, electric_field: float = 3.0, temperature: float = 300.) -> np.ndarray:
        """
        Return the stored custom energy matrix array without transformation.

        Parameters
        ----------
        fermi_level : float, default 9.5
            Ignored parameters included for subclass compatibility.
        work_function : float, default 4.5
            Ignored parameters included for subclass compatibility.
        above_barrier : float, default 10
            Ignored parameters included for subclass compatibility.
        electric_field : float, default 3.0
            Ignored parameters included for subclass compatibility.
        temperature : float, default 300.0
            Ignored parameters included for subclass compatibility.

        Returns
        -------
        numpy.ndarray
            The original initialization array reference map context.

        Examples
        --------
        >>> array = np.array([1, 2, 3])
        >>> cm = CustomMetal(array)
        >>> cm.generate_band_structure()
        array([1, 2, 3])
        """
        return self.custom_energy_array