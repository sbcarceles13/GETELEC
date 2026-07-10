"""
Electron Emission Simulation Module.

This module provides modeling classes for electron field and thermal emission 
from both metallic and semiconductor surfaces. It integrates modules for potential 
barrier profile evaluation, transmission probability solvers, electron supply functions, 
and band structure setups to compute critical performance metrics such as 
current density, energy distributions, and Nottingham effect heating.

Dependencies
------------
numpy : Array manipulation and vectorized math.
scipy.integrate.cumulative_trapezoid : Integration helper for energy distributions.
getelec : Domain-specific package containing structural definitions and physical constants.
"""

import numpy as np
from getelec import constants as const
from getelec import potential_barrier
from getelec import transmission_solver
from getelec import electron_supply
from getelec import band_structure
from getelec.constants import Q
from scipy.integrate import cumulative_trapezoid

class MetalEmitter:
    """
    Evaluates electron emission characteristics for metallic materials.

    This class coordinates the sub-modules responsible for calculating electron emission 
    current densities, total energy distributions (TED), normal energy distributions (NED), 
    and Nottingham heat contributions for metals under given electronic or thermal setups.

    Parameters
    ----------
    potential : potential_barrier.Barrier
        The potential barrier object defining work functions, fields, and levels.
    solver : transmission_solver.TransmissionSolver
        The solver utilized to evaluate transmission probabilities across barriers.
    supply : electron_supply.Supply
        The supply function calculation object relating temperature and energy states.
    band : band_structure.BandStructure
        The electronic band configuration and sampling grid resolution controls.

    Attributes
    ----------
    potential : potential_barrier.Barrier
        Stored reference to the barrier properties.
    solver : transmission_solver.TransmissionSolver
        Stored reference to the numerical transmission evaluator.
    supply : electron_supply.Supply
        Stored reference to the underlying supply functionality.
    band : band_structure.BandStructure
        Stored reference to the grid geometry and material parameters.
    """

    def __init__(self, 
                 potential: 'potential_barrier.Barrier', 
                 solver: 'transmission_solver.TransmissionSolver', 
                 supply: 'electron_supply.Supply', 
                 band: 'band_structure.BandStructure'):
        """
        Initialize the MetalEmitter with structural and numerical solver engines.
        """
        self.potential = potential
        self.solver = solver
        self.supply = supply
        self.band = band

    def update_params(self, **kwargs):
        """
        Broadcasting parameter updates to sub-modules.

        Maps simplified parameter aliases to actual attribute keys across child instances 
        (potential, supply, band, solver), allowing programmatic sweeps or optimizations 
        without explicit multi-object assignments.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments representing parameters to update. Common keys include:
            - 'field' or 'electric_field' : float
            - 'temp' or 'temperature' : float
            - 'wf' or 'work_function' : float
            - 'fermi' or 'fermi_level' : float
            - 'radius' : float
            - 'gamma' : float
            - 'lower_energy_limit' : float
            - 'upper_energy_limit' : float
            - 'energy_resolution' : float

        Returns
        -------
        None

        Examples
        --------
        >>> emitter.update_params(electric_field=4.0, temperature=1000)
        >>> emitter.update_params(temp=300, wf=4.5)
        """
        # Define common aliases to make curve_fit code more readable
        alias_map = {
            'field': 'electric_field',
            'temp': 'temperature',
            'wf': 'work_function',
            'fermi': 'fermi_level',
            'radius' : 'radius',
            'gamma' : 'gamma',
            'lower_energy_limit' : 'lower_energy_lim',
            'upper_energy_limit' : 'upper_energy_lim',
            'energy_resolution' : 'energy_resolution'
        }

        for key, value in kwargs.items():
            # Resolve aliases (e.g., convert 'temp' to 'temperature')
            target_key = alias_map.get(key, key)

            # 1. Update Potential (electric_field, work_function, fermi_level)
            if hasattr(self.potential, target_key):
                setattr(self.potential, target_key, value)

            # 2. Update Supply (temperature, fermi_level)
            if hasattr(self.supply, target_key):
                setattr(self.supply, target_key, value)

            # 3. Update Grid (if it has specific params like energy_resolution)
            if hasattr(self.band, target_key):
                setattr(self.band, target_key, value)

            if hasattr(self.solver, target_key):
                setattr(self.solver, target_key, value)

    def _calculate_base_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Private helper to compute grid, supply, and transmission.

        Evaluates the core physical vectors required for emission integrals based on 
        the active parameters of linked dependencies.

        Returns
        -------
        energies : numpy.ndarray
            1D array containing computed energy states sampled along the profile.
        supply_vals : numpy.ndarray
            1D array containing electron supply factors corresponding to each energy state.
        trans_vals : numpy.ndarray
            1D array containing transmission probabilities through the potential barrier.

        Raises
        ------
        AttributeError
            If an underlying instance object does not implement required interface methods.
        """
        # 1. Create Energy Grid
        energies = self.band.generate_band_structure(
            fermi_level=self.potential.fermi_level, 
            work_function=self.potential.work_function, 
            above_barrier=2,  
            electric_field=self.potential.electric_field, 
            temperature=self.supply.temperature
        )
        
        supply_vals = self.supply.get_supply(energies)
    
        trans_vals = self.solver.calculate_transmission(self.potential, energies)

        return energies, supply_vals, trans_vals

    def calculate_current_density(self) -> float:
        """
        Calculates total Current Density.

        Integrates the product of electron supply and transmission profiles across 
        the metal's active band structure energy range, scaling with physical factors 
        to report macroscopic dimensions.

        Returns
        -------
        current_density : float
            The resulting total current density, typically units normalized to A/cm^2.

        Examples
        --------
        >>> j = metal_emitter.calculate_current_density()
        >>> print(f"Current density: {j} A/cm2")
        """
        energies, supply_vals, trans_vals = self._calculate_base_data()
        
        # dE = energies[1] - energies[0]

        # cum_trans_vals = cumulative_trapezoid(trans_vals, energies, initial=0)
        
        # # Corrects the initial zero-value artifact from cumulative_trapezoid. 
        # # Approximating the low-energy tail exponentially provides an analytical correction, 
        # # avoiding the overhead of slower quad integration.
        # if trans_vals[0] > 1e-100:

        #     k = (np.log(trans_vals[1]) - np.log(trans_vals[0])) / dE

        #     tail_correction = trans_vals[0] / k

        #     cum_trans_vals += tail_correction

        # integ = cum_trans_vals * trans_vals
        # current_density = np.sum(integ) * self.band.energy_resolution * self.supply.temperature * const.J_PREFACTOR_CONST * 1E14 * 1E15 * Q # Factor for /nm2 --> /cm2, sf --> s, and from electrons --> charge

        integ = supply_vals * trans_vals
        current_density = np.sum(integ) * self.band.energy_resolution * self.supply.temperature * const.J_PREFACTOR_CONST * 1E14 * 1E15 * Q # Factor for /nm2 --> /cm2, sf --> s, and from electrons --> charge
        return current_density 
    
    def calculate_total_energy_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates Total Energy Distribution (TED).

        Computes the energy spectral intensity utilizing cumulative trapezoidal integration 
        along the transmission spectra combined with low-energy exponential tail analytical 
        corrections to optimize speed and accuracy.

        Returns
        -------
        energies : numpy.ndarray
            The 1D array representing sampling energy coordinates.
        ted : numpy.ndarray
            The 1D array representing total energy distribution values.

        Raises
        ------
        IndexError
            If generated energy arrays contain fewer than two grid positions.
        """
        energies, supply_vals, trans_vals = self._calculate_base_data()
        
        dE = energies[1] - energies[0]

        cum_trans_vals = cumulative_trapezoid(trans_vals, energies, initial=0)
        
        # Corrects the initial zero-value artifact from cumulative_trapezoid. 
        # Approximating the low-energy tail exponentially provides an analytical correction, 
        # avoiding the overhead of slower quad integration.
        if trans_vals[0] > 1e-100:

            k = (np.log(trans_vals[1]) - np.log(trans_vals[0])) / dE

            tail_correction = trans_vals[0] / k

            cum_trans_vals += tail_correction

        ted = supply_vals * cum_trans_vals * const.J_PREFACTOR_CONST * 1E14 * 1E15 * Q # Factor for /nm2 --> /cm2, sf --> s, and from electrons --> charge

        # max_val = np.max(ted)
        # if max_val > 0:
        #     return energies, ted# / max_val
        return energies, ted
    
    def calculate_normal_energy_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates Normal Energy Distribution (NED).

        Evaluates emission distribution strictly normalized along the surface-orthogonal vector, 
        calculated directly from point-wise product mappings of the energy state components.

        Returns
        -------
        energies : numpy.ndarray
            The 1D array representing sampling energy coordinates.
        ned : numpy.ndarray
            The 1D array representing normal energy distribution values.
        """
        energies, supply_vals, trans_vals = self._calculate_base_data()
        
        ned = supply_vals * trans_vals * const.J_PREFACTOR_CONST * 1E14 * 1E15 * Q # Factor for /nm2 --> /cm2, sf --> s, and from electrons --> charge

        return energies, ned
    
    def calculate_nottingham_heat(self) -> float:
        """
        Calculates the Nottingham heat.

        Measures energy exchange variations near the emission surface by computing the 
        average energy carried away by emitted electrons relative to the active Fermi level.

        Returns
        -------
        nottingham_heat : float
            The weighted average exchange energy component. Returns 0.0 if the cumulative 
            total emission current yields zero.
        """

        energies, ted = self.calculate_total_energy_distribution()

        total_current = self.calculate_current_density()

        weighted_energy = np.trapezoid((energies-self.potential.fermi_level) * ted, energies)

        if total_current > 0:
            return weighted_energy / total_current
        return 0.0

class SemiconductorEmitter:
    """
    Evaluates electron emission characteristics for semiconductor substrates.

    This class decouples and manages conduction band (CB) and valence band (VB) multi-band 
    contributions to simulate field and thermal electron emission from semiconductor profiles, 
    accounting for distinct electron and hole effective masses.

    Parameters
    ----------
    potential : potential_barrier.Barrier
        The potential barrier object defining work functions, fields, and levels.
    solver : transmission_solver.TransmissionSolver
        The solver utilized to evaluate transmission probabilities across barriers.
    supply : electron_supply.Supply
        The supply function calculation object relating temperature and energy states.
    band : band_structure.BandStructure
        The semiconductor electronic band configuration containing effective masses 
        and structural descriptors.

    Attributes
    ----------
    potential : potential_barrier.Barrier
        Stored reference to the barrier properties.
    solver : transmission_solver.TransmissionSolver
        Stored reference to the numerical transmission evaluator.
    supply : electron_supply.Supply
        Stored reference to the underlying supply functionality.
    band : band_structure.BandStructure
        Stored reference to the semiconductor band boundaries and state models.
    """

    def __init__(self, 
                 potential: 'potential_barrier.Barrier', 
                 solver: 'transmission_solver.TransmissionSolver', 
                 supply: 'electron_supply.Supply', 
                 band: 'band_structure.BandStructure'):
        """
        Initialize the SemiconductorEmitter with structural and numerical solver engines.
        """
        self.potential = potential
        self.solver = solver
        self.supply = supply
        self.band = band

    def update_params(self, **kwargs):
        """
        Broadcasting parameter updates to sub-modules.

        Maps simplified parameter aliases to actual attribute keys across child instances 
        (potential, supply, band, solver), allowing programmatic sweeps or optimizations 
        without explicit multi-object assignments.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments representing parameters to update. Common keys include:
            - 'field' or 'electric_field' : float
            - 'temp' or 'temperature' : float
            - 'wf' or 'work_function' : float
            - 'fermi' or 'fermi_level' : float
            - 'radius' : float
            - 'gamma' : float
            - 'lower_energy_limit' : float
            - 'upper_energy_limit' : float
            - 'energy_resolution' : float

        Returns
        -------
        None

        Examples
        --------
        >>> semiconductor_emitter.update_params(field=5.0, temp=400)
        """
        # Define common aliases to make curve_fit code more readable
        alias_map = {
            'field': 'electric_field',
            'temp': 'temperature',
            'wf': 'work_function',
            'fermi': 'fermi_level',
            'radius' : 'radius',
            'gamma' : 'gamma',
            'lower_energy_limit' : 'lower_energy_lim',
            'upper_energy_limit' : 'upper_energy_lim',
            'energy_resolution' : 'energy_resolution'
        }

        for key, value in kwargs.items():
            # Resolve aliases (e.g., convert 'temp' to 'temperature')
            target_key = alias_map.get(key, key)

            # 1. Update Potential (electric_field, work_function, fermi_level)
            if hasattr(self.potential, target_key):
                setattr(self.potential, target_key, value)

            # 2. Update Supply (temperature, fermi_level)
            if hasattr(self.supply, target_key):
                setattr(self.supply, target_key, value)

            # 3. Update Grid (if it has specific params like energy_resolution)
            if hasattr(self.band, target_key):
                setattr(self.band, target_key, value)

            if hasattr(self.solver, target_key):
                setattr(self.solver, target_key, value)

    def _calculate_base_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Private helper to compute grid, supply, and transmission.

        Resolves sub-band structures for both Conduction and Valence bands, evaluates individual 
        transmission indices, and uses respective charge carrier effective masses to weight 
        and calculate final cumulative transmission arrays.

        Returns
        -------
        eecb1 : numpy.ndarray
            Conduction band energy tracking grid.
        eevb1 : numpy.ndarray
            Valence band energy tracking grid.
        supply_cb : numpy.ndarray
            Electron supply vector within Conduction band states.
        supply_vb : numpy.ndarray
            Electron supply vector within Valence band states.
        trans_cb : numpy.ndarray
            Effective conduction band transmission probability after applying electron mass weighting.
        trans_vb : numpy.ndarray
            Effective valence band transmission probability after applying hole mass weighting.
        """
        eecb1, eecb2, eevb1, eevb2 = self.band.generate_band_structure(fermi_level=self.potential.fermi_level, work_function=self.potential.work_function, above_barrier=2, electric_field=self.potential.electric_field, temperature=self.supply.temperature)
    
        supply_cb = self.supply.get_supply(eecb1)
        supply_vb = self.supply.get_supply(eevb1)
    
        trans_cb_1 = self.solver.calculate_transmission(self.potential, eecb1)
        trans_cb_2 = self.solver.calculate_transmission(self.potential, eecb2)
        trans_vb_1 = self.solver.calculate_transmission(self.potential, eevb1)
        trans_vb_2 = self.solver.calculate_transmission(self.potential, eevb2)

        trans_cb = trans_cb_1 - (1 - self.band.electron_eff_mass) * trans_cb_2
        trans_vb = trans_vb_1 - (1 + self.band.hole_eff_mass) * trans_vb_2

        return eecb1, eevb1, supply_cb, supply_vb, trans_cb, trans_vb

    def calculate_current_density(self) -> float:
        """
        Calculates total Current Density.

        Computes independent current density terms for both Conduction and Valence band channels 
        by taking discrete integration over state properties and summarizes them into a collective output.

        Returns
        -------
        current_density : float
            Total macro-scaled current density across active bands (A/cm^2).
        """
        energies_cb, energies_vb, supply_cb, supply_vb, trans_cb, trans_vb = self._calculate_base_data()
        
        integ = supply_cb * trans_cb
        current_density_cb = np.sum(integ) * self.band.energy_resolution * self.supply.temperature * const.J_PREFACTOR_CONST * 1E14 * 1E15 * Q # Factor for /nm2 --> /cm2, sf --> s, and from electrons --> charge

        integ = supply_vb * trans_vb
        current_density_vb = np.sum(integ) * self.band.energy_resolution * self.supply.temperature * const.J_PREFACTOR_CONST * 1E14 * 1E15 * Q # Factor for /nm2 --> /cm2, sf --> s, and from electrons --> charge
        
        current_density = current_density_cb + current_density_vb

        return current_density
    
    def calculate_total_energy_distribution(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates Total Energy Distribution (TED).

        Evaluates individual total energy spectral slices across conduction and valence matrices. 
        Includes analytical low-energy boundary modifications to offset cumulative numerical tracking noise.

        Returns
        -------
        energies_cb : numpy.ndarray
            Conduction band energy positions.
        ted_cb : numpy.ndarray
            Conduction band energy distribution values.
        energies_vb : numpy.ndarray
            Valence band energy positions.
        ted_vb : numpy.ndarray
            Valence band energy distribution values.

        Raises
        ------
        IndexError
            If any extracted internal band configurations map to fewer than two discrete points.
        """
        energies_cb, energies_vb, supply_cb, supply_vb, trans_cb, trans_vb = self._calculate_base_data()
        
        dE = energies_cb[1] - energies_cb[0]

        cum_trans_vals = cumulative_trapezoid(trans_cb, energies_cb, initial=0)
        
        ted_cb = supply_cb * cum_trans_vals * const.J_PREFACTOR_CONST * 1E14 * 1E15 * Q # Factor for /nm2 --> /cm2, sf --> s, and from electrons --> charge

        dE = energies_vb[1] - energies_vb[0]

        cum_trans_vals = cumulative_trapezoid(trans_vb, energies_vb, initial=0)
        
        # Corrects the initial zero-value artifact from cumulative_trapezoid. 
        # Approximating the low-energy tail exponentially provides an analytical correction, 
        # avoiding the overhead of slower quad integration.
        if trans_vb[0] > 1e-100:

            k = (np.log(trans_vb[1]) - np.log(trans_vb[0])) / dE

            tail_correction = trans_vb[0] / k

            cum_trans_vals += tail_correction

        ted_vb = supply_vb * cum_trans_vals * const.J_PREFACTOR_CONST * 1E14 * 1E15 * Q # Factor for /nm2 --> /cm2, sf --> s, and from electrons --> charge

        return energies_cb, ted_cb, energies_vb, ted_vb
    
    def calculate_normal_energy_distribution(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates Normal Energy Distribution (NED).

        Extracts multi-band energy profile distributions focused parallel to the emission vector 
        orthogonal to the surface boundary layer.

        Returns
        -------
        energies_cb : numpy.ndarray
            Conduction band energy coordinates.
        ned_cb : numpy.ndarray
            Conduction band normal distribution weights.
        energies_vb : numpy.ndarray
            Valence band energy coordinates.
        ned_vb : numpy.ndarray
            Valence band normal distribution weights.
        """
        energies_cb, energies_vb, supply_cb, supply_vb, trans_cb, trans_vb = self._calculate_base_data()
        
        ned_cb = supply_cb * trans_cb * const.J_PREFACTOR_CONST * 1E14 * 1E15 * Q # Factor for /nm2 --> /cm2, sf --> s, and from electrons --> charge

        ned_vb = supply_vb * trans_vb * const.J_PREFACTOR_CONST * 1E14 * 1E15 * Q # Factor for /nm2 --> /cm2, sf --> s, and from electrons --> charge

        return energies_cb, ned_cb, energies_vb, ned_vb
    
    def calculate_nottingham_heat(self) -> float:
        """
        Calculates the Nottingham heat.

        Measures energy dissipation or structural cooling across conduction and valence pathways, 
        offsetting results using collective semiconductor transport coefficients.

        Returns
        -------
        nottingham_heat : float
            Combined multi-band exchange heat term. Returns 0.0 if total combined 
            current returns zero.
        """

        energies_cb, ted_cb, energies_vb, ted_vb = self.calculate_total_energy_distribution()

        total_current = self.calculate_current_density()

        weighted_energy_cb = np.trapezoid((energies_cb-self.potential.fermi_level) * ted_cb, energies_cb)
        weighted_energy_vb = np.trapezoid((energies_vb-self.potential.fermi_level) * ted_vb, energies_vb)

        weighted_energy = weighted_energy_cb + weighted_energy_vb

        if total_current > 0:
            return weighted_energy / total_current
        return 0.0
