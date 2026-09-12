"""
Band Structure Calculation Module.

This module provides a suite of classes to model and generate energy grids and
band structures for metals and semiconductors: fixed grids, grids trimmed
automatically to the energies that carry current ('smart'), and grids supplied
by the user ('custom').

"""

import warnings

import numpy as np
from abc import ABC, abstractmethod
from getelec import constants as const


def _generate_band_grids(lower_energy_lim, top_valence, band_gap, upper_energy_lim, energy_resolution):
    """
    Uniform energy grids for the conduction and valence bands, in eV.

    The conduction band runs up from its edge ``E_C = E_V + E_g``, the valence
    band up from ``lower_energy_lim`` to its top ``E_V``. Both use the published
    ``arange`` convention, so the valence grid can end up to one step above
    ``E_V``; the distributions clamp that point to zero.
    """
    bottom_conduction = top_valence + band_gap
    conduction = np.arange(bottom_conduction, upper_energy_lim + energy_resolution, energy_resolution)
    valence = np.arange(lower_energy_lim, top_valence + energy_resolution, energy_resolution)
    return conduction, valence


def _get_window_limits(conduction, valence, top_valence, band_gap, electron_eff_mass, hole_eff_mass):
    """
    The band grids together with the lower limit of each point's E_z window.

    Returns
    -------
    tuple of numpy.ndarray
        ``(conduction, conduction_lower, valence, valence_lower)``: each band's
        energies and, for every energy E, the lower limit of the normal-energy
        window ``[lower(E), E]`` its distributions integrate D over.
    """
    bottom_conduction = top_valence + band_gap
    # Eq. (8) of the paper, and A.16/A.21 of its appendix: the transverse
    # energy is bounded by E_r <= (m*/m) x (kinetic energy in the band), so
    # the E_z window runs from E down to
    #
    #     E - (m_e*/m)(E - E_C)  =  E_C + (1 - m_e*/m)(E - E_C) ,
    #
    # i.e. alpha_bar = 1 - m*/m measured from the band edge. Using alpha
    # instead of alpha_bar makes the window (m*/m)/(1 - m*/m) times too
    # narrow -- 49x for m* = 0.98 -- which collapses the conduction TED onto
    # the shape of the NED, and puts the lower limit *above* E entirely once
    # m* > m, so the band disappears. This mirrors the valence expression
    # below, which is E - (m_h*/m)(E_V - E).
    conduction_lower = conduction - electron_eff_mass * (conduction - bottom_conduction)
    valence_lower = ((1+hole_eff_mass) * valence) - (hole_eff_mass * top_valence)
    return conduction, conduction_lower, valence, valence_lower


def _get_barrier_width_energy(fermi_level, work_function, electric_field, barrier_width):
    """
    Energy below which the Schottky-Nordheim barrier is wider than ``barrier_width``, in eV.

    The criterion :class:`SmartMetal` uses for the bottom of its grid. The
    turning points at energy E are the roots of ``F x^2 - (W - E) x + k_e/4``,
    ``W = E_F + phi``, so the barrier is ``sqrt((W - E)^2 - k_e F) / F`` wide.
    """
    total = fermi_level + work_function
    if electric_field <= 0:
        return total
    return total - np.sqrt((electric_field*barrier_width)**2+(4*electric_field*const.IMAGE_PREFACTOR))


def _get_supply_threshold_energy(fermi_level, temperature, supply_threshold):
    """
    Energy above which the supply ``ln(1 + exp(-(E - E_F)/k_B T))`` falls below ``supply_threshold``, in eV.

    The criterion :class:`SmartMetal` uses for the top of its grid. At zero
    temperature the occupancy is a step, and the grid ends at the Fermi level.
    """
    if supply_threshold <= 0:
        raise ValueError(f"supply_threshold must be positive, got {supply_threshold}.")
    if temperature < 1e-6:
        return fermi_level
    return fermi_level - const.KB * temperature * np.log(np.expm1(supply_threshold))


def _trim_grid(grid, start, end):
    """
    The points of a uniform, increasing grid that cover ``[start, end]``.

    One point beyond each end is kept, so the trapezoid rule still reaches both
    limits. Empty when fewer than two points would survive: a band that lies
    entirely outside the window cannot emit.
    """
    if grid.size < 2 or end < start:
        return grid[:0]
    step = grid[1] - grid[0]
    first = max(int(np.ceil((start - grid[0]) / step - 1e-9)) - 1, 0)
    last = min(int(np.floor((end - grid[0]) / step + 1e-9)) + 1, grid.size - 1)
    if last <= first:
        return grid[:0]
    return grid[first:last + 1]


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
    def generate_band_structure(self, fermi_level: float, work_function: float, electric_field: float, temperature: float) -> np.ndarray:
        """
        Generate the energy grid or band structure for the material.

        Parameters
        ----------
        fermi_level : float
            The Fermi energy level of the system, in electron-volts (eV)
        work_function : float
            The work function of the material, in electron-volts (eV)
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

    def get_state_weights(self, energy_array: np.ndarray) -> np.ndarray:
        """
        How many states each total energy carries, relative to a free electron gas.

        The current integral of the paper counts the states available at total
        energy E through the inner integral over normal energy, ``integral of D
        dE_z`` across the window, which for a parabolic band is the area of the
        k-parallel disc. A band structure that knows better than the free
        electron model returns the ratio here, and the emitter multiplies the
        total energy distribution by it; the normal energy distribution is then
        rebuilt from the same double integral, so the two still integrate to the
        same current density.

        The base class returns ones, which is the free electron model and the
        behaviour of every band structure that does not override this.

        Parameters
        ----------
        energy_array : numpy.ndarray
            Total energies, in eV.

        Returns
        -------
        numpy.ndarray
            Dimensionless weights, the same shape as ``energy_array``.
        """
        return np.ones_like(np.asarray(energy_array, dtype=float))

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
    energy_resolution : float, default 0.002
        The discretization step size for the base grids. Finer than a metal's
        default because each band's grid ends at its band edge, where the TED
        has a finite slope: the trapezoid error in the current is then
        ~(h / k_B T)^2 / 12, about 1.1% at 0.01 eV and 300 K.

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
    def __init__(self, lower_energy_limit: float = 0.001, top_valence: float = 12.5, fermi_level: float = 13.0, band_gap: float = 1.12, electron_eff_mass = 1.64, hole_eff_mass = 0.68,  upper_energy_limit: float = 30.0, energy_resolution: float = 0.002):
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
        energies_conduction_band1, energies_valence_band1 = _generate_band_grids(
            self.lower_energy_lim, self.top_valence, self.band_gap,
            self.upper_energy_lim, self.energy_resolution)
        return _get_window_limits(energies_conduction_band1, energies_valence_band1,
                                  self.top_valence, self.band_gap,
                                  self.electron_eff_mass, self.hole_eff_mass)

class SmartMetal(BandStructure):
    """
    Represents an adaptive metal grid system optimizing energy boundaries dynamically.

    Notes
    -----
    Use only with SN barrier. The rest of the barriers need their own optimisation protocol.
    If not SN barrier, use Metal or Custom.

    The defaults are built for the currents an emitter actually produces. They
    hold the current density and the Nottingham heat within 1% of a far wider
    grid wherever the current density reaches 1e-12 A/cm^2, which is 1 pA from
    1 cm^2 -- checked for the planar, sharp-tip and triangular barriers, work
    functions of 2.5 to 6 eV, fields of 0.2 to 12 V/nm and 300 to 3000 K.
    Below 1 pA/cm^2 the cuts start to show, and the smaller the current the
    more it costs to reach: the same 1% at arbitrarily small currents would
    need a ``supply_threshold`` around 1e-40 and the energies to match. If
    those currents are what you are after, pass a smaller ``supply_threshold``
    and a larger ``barrier_width``, or set the limits yourself with
    :class:`Metal` or :class:`CustomMetal`.

    Parameters
    ----------
    barrier_width : float, default 3.0
        The spatial width of the potential barrier, beyond which there is effectivively no tunneling.
    supply_threshold : float, default 5e-24
        The numerical threshold constraint used to cap maximum grid energies, beyond which the emitted electrons do not substatially contribute to the total emitted current.
        The cut sits ``k_B T * ln(1 / supply_threshold)`` above the Fermi level, 53.7 k_B T at the default. Thermionic emission comes from the top of the barrier, so the grid has to reach it: at 32.2 k_B T (a threshold of 1e-14) the cut falls below the top once the top lies more than about 25 k_B T above the Fermi level, and the current is then underestimated -- by 7% for a 4.5 eV work function at 0.5 V/nm and 1500 K.
    energy_resolution : float, default 0.01
        The structural resolution step size of the resulting mesh.

    Attributes
    ----------
    barrier_width : float
        The spatial width of the potential barrier.
    supply_threshold : float
        The numerical threshold constraint.
    """
    def __init__(self, barrier_width: float = 3.0, supply_threshold: float = 5e-24, energy_resolution: float = 0.01):
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

    def generate_band_structure(self, fermi_level: float = 9.5, work_function: float = 4.5, electric_field: float = 3.0, temperature: float = 300.) -> np.ndarray:
        """
        Generate an optimized dynamic array grid bounded by the calculated limits.

        Parameters
        ----------
        fermi_level : float, default 9.5
            The structural Fermi energy level.
        work_function : float, default 4.5
            The target material work function.
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
        # At low field the barrier is wider than ``barrier_width`` even at the
        # Fermi level, so that criterion alone starts the grid above its end and
        # leaves it empty -- current_density(field=1.0) used to fail at room
        # temperature. Field emission comes from within a few decay widths of
        # E_F (Young's transverse energy d, ~0.05 eV per V/nm, although the
        # apparent width varies with energy: Barranco Carceles et al., IVNC
        # 2026, doi:10.1109/IVNC69421.2026.11660967), so the grid never starts
        # higher than 1 eV below it: J then agrees with a far wider grid to
        # better than 1e-5 at 1 V/nm. From ~2 V/nm up the barrier criterion is
        # already lower, so nothing changes there.
        fermi_floor = np.floor((fermi_level - 1.0) / self.energy_resolution) * self.energy_resolution
        energy_low = max(min(energy_low, fermi_floor), self.energy_resolution)
        energy_high = self._get_max_energy(fermi_level,self.energy_resolution,temperature)
        return np.arange(energy_low, energy_high + self.energy_resolution, self.energy_resolution)

class SmartSemiconductor(BandStructure):
    """
    Semiconductor band grids trimmed with the two criteria of :class:`SmartMetal`.

    The grids of :class:`Semiconductor`, cut in each band:

    - **bottom**: where the Schottky-Nordheim barrier becomes wider than
      ``barrier_width``. As in ``SmartMetal``, a band's grid never starts
      higher than 1 eV below the top of its occupied states,
      ``min(E_F, band top)``: field emission comes from within a few decay
      widths of that energy, and at low field the barrier criterion alone would
      start above it. For the conduction band that is 1 eV below ``E_F``; for
      the valence band 1 eV below ``min(E_F, E_V)``, so a valence band lying
      more than 1 eV below the Fermi level keeps the energies just below ``E_V``
      it emits from.
    - **top**: where the supply ``ln(1 + exp(-(E - E_F)/k_B T))`` falls below
      ``supply_threshold``.

    What survives is clipped to each band: the conduction grid never starts
    below ``E_C``, and the valence grid never runs past the top of the
    ``Semiconductor`` valence grid. A band that lies entirely outside the window
    comes back empty and contributes nothing -- the conduction band of a
    semiconductor at 0 K with the Fermi level in the gap, for example. The
    points kept are exactly those of the ``Semiconductor`` grids with the same
    parameters, so the two agree wherever neither cut applies.

    Parameters
    ----------
    lower_energy_limit : float, default 0.001
        The bottom of the valence band, and the zero of the energy scale, in eV.
    top_valence : float, default 12.5
        The top of the valence band, in eV.
    fermi_level : float, default 13.0
        The Fermi level, in eV. Used when ``generate_band_structure`` is not
        given one; an emitter passes its barrier's.
    band_gap : float, default 1.12
        The band gap, in eV.
    electron_eff_mass : float, default 1.64
        Conduction band effective mass, relative to the free electron mass.
    hole_eff_mass : float, default 0.68
        Valence band effective mass, relative to the free electron mass.
    upper_energy_limit : float, default 30.0
        The top of the conduction band grid before trimming, in eV.
    energy_resolution : float, default 0.002
        The grid step, in eV. See :class:`Semiconductor` for why it is finer
        than a metal's.
    barrier_width : float, default 3.0
        The barrier width, in nm, below whose energy the transmission is taken
        as negligible.
    supply_threshold : float, default 5e-24
        The supply below which the electrons are taken as negligible. The cut
        sits ``k_B T * ln(1 / supply_threshold)`` above the Fermi level, 53.7
        ``k_B T`` at the default, which keeps the barrier top inside the grid
        wherever the current reaches 1e-12 A/cm^2. See :class:`SmartMetal`.

    Attributes
    ----------
    lower_energy_lim, upper_energy_lim, energy_resolution : float
        As for :class:`Semiconductor`.
    top_valence, fermi_level, band_gap : float
        As for :class:`Semiconductor`.
    electron_eff_mass, hole_eff_mass : float
        As for :class:`Semiconductor`.
    barrier_width, supply_threshold : float
        As for :class:`SmartMetal`.

    Notes
    -----
    The barrier criterion uses the Schottky-Nordheim barrier, as ``SmartMetal``
    does, whatever barrier the emitter holds.

    As for :class:`SmartMetal`, the defaults are set for currents down to
    1e-12 A/cm^2 (1 pA from 1 cm^2). For currents far below that, pass a
    smaller ``supply_threshold`` and a larger ``barrier_width``, or give the
    grids yourself with :class:`CustomSemiconductor`.
    """
    def __init__(self, lower_energy_limit: float = 0.001, top_valence: float = 12.5, fermi_level: float = 13.0, band_gap: float = 1.12, electron_eff_mass = 1.64, hole_eff_mass = 0.68,  upper_energy_limit: float = 30.0, energy_resolution: float = 0.002, barrier_width: float = 3.0, supply_threshold: float = 5e-24):
        """
        Initialize the SmartSemiconductor instance with band and trimming parameters.
        """
        super().__init__(energy_resolution=energy_resolution)

        self.lower_energy_lim = lower_energy_limit
        self.upper_energy_lim = upper_energy_limit
        self.top_valence = top_valence
        self.fermi_level = fermi_level
        self.band_gap = band_gap
        self.electron_eff_mass = electron_eff_mass
        self.hole_eff_mass = hole_eff_mass
        self.barrier_width = barrier_width
        self.supply_threshold = supply_threshold

    def generate_band_structure(self, fermi_level: float = None, work_function: float = 4.5, electric_field: float = 3.0, temperature: float = 300.) -> tuple:
        """
        Generate the trimmed conduction and valence grids and their window limits.

        Parameters
        ----------
        fermi_level : float, optional
            The Fermi level, in eV. Defaults to the instance's own.
        work_function : float, default 4.5
            The work function, in eV.
        electric_field : float, default 3.0
            The field, in V/nm.
        temperature : float, default 300.0
            The temperature, in K.

        Returns
        -------
        tuple of numpy.ndarray
            The same four arrays as :meth:`Semiconductor.generate_band_structure`:
            each band's energies and the lower limit of its E_z window. Either
            band may be empty.

        Examples
        --------
        >>> smart = SmartSemiconductor(energy_resolution=0.01)
        >>> ec1, ec2, ev1, ev2 = smart.generate_band_structure(work_function=4.5, electric_field=5.0, temperature=300.0)
        """
        fermi = self.fermi_level if fermi_level is None else fermi_level
        conduction, valence = _generate_band_grids(
            self.lower_energy_lim, self.top_valence, self.band_gap,
            self.upper_energy_lim, self.energy_resolution)

        energy_low = _get_barrier_width_energy(fermi, work_function, electric_field, self.barrier_width)
        energy_high = _get_supply_threshold_energy(fermi, temperature, self.supply_threshold)

        # The low-field floor of SmartMetal, taken from the top of each band's
        # occupied states: E_F for the conduction band, min(E_F, E_V) for the
        # valence band. Taken from E_F alone, a valence band more than 1 eV
        # below the Fermi level would be cut away entirely at low field.
        conduction = _trim_grid(conduction, min(energy_low, fermi - 1.0), energy_high)
        valence = _trim_grid(valence, min(energy_low, min(fermi, self.top_valence) - 1.0), energy_high)

        return _get_window_limits(conduction, valence, self.top_valence, self.band_gap,
                                  self.electron_eff_mass, self.hole_eff_mass)

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

    def generate_band_structure(self, fermi_level: float = 9.5, work_function: float = 4.5, electric_field: float = 3.0, temperature: float = 300.) -> np.ndarray:
        """
        Return the stored custom energy matrix array without transformation.

        Parameters
        ----------
        fermi_level : float, default 9.5
            Ignored parameters included for subclass compatibility.
        work_function : float, default 4.5
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

class CustomSemiconductor(BandStructure):
    """
    A semiconductor band structure on energy grids the user supplies, one per band.

    The counterpart of :class:`CustomMetal`: the energies are returned as given,
    for a range, a resolution or a spacing that :class:`Semiconductor` does not
    offer. What a semiconductor needs beyond the energies -- the lower limit of
    each energy's E_z window, Eq. (8) of the paper -- follows from the band
    edges and effective masses, exactly as in ``Semiconductor``, so the physics
    does not depend on which of the two built the grid.

    The grids are checked when the band structure is generated, because points
    outside a band have no states and would make the distributions
    inconsistent: each must be a finite, strictly increasing 1-D array, the
    conduction energies must not lie below ``E_C = top_valence + band_gap``, and
    the valence energies must not lie above ``top_valence``. Either band may be
    empty, and then contributes nothing.

    Parameters
    ----------
    custom_conduction_array : array_like
        Conduction band energies, in eV.
    custom_valence_array : array_like
        Valence band energies, in eV.
    top_valence : float, default 12.5
        The top of the valence band, in eV.
    band_gap : float, default 1.12
        The band gap, in eV.
    electron_eff_mass : float, default 1.64
        Conduction band effective mass, relative to the free electron mass.
    hole_eff_mass : float, default 0.68
        Valence band effective mass, relative to the free electron mass.

    Attributes
    ----------
    custom_conduction_array, custom_valence_array : numpy.ndarray
        The energies of each band.
    top_valence, band_gap, electron_eff_mass, hole_eff_mass : float
        As for :class:`Semiconductor`.

    Notes
    -----
    The distributions are verified on uniform grids, which is what the other
    band structures produce. A non-uniform grid is integrated correctly, but
    the transmission over each window is sampled at the grid's first step.
    """
    def __init__(self, custom_conduction_array: np.ndarray, custom_valence_array: np.ndarray, top_valence: float = 12.5, band_gap: float = 1.12, electron_eff_mass: float = 1.64, hole_eff_mass: float = 0.68):
        """
        Initialize the CustomSemiconductor instance with its band grids and parameters.
        """
        self.custom_conduction_array = np.asarray(custom_conduction_array, dtype=float)
        self.custom_valence_array = np.asarray(custom_valence_array, dtype=float)
        self.top_valence = top_valence
        self.band_gap = band_gap
        self.electron_eff_mass = electron_eff_mass
        self.hole_eff_mass = hole_eff_mass

    def generate_band_structure(self, fermi_level: float = 9.5, work_function: float = 4.5, electric_field: float = 3.0, temperature: float = 300.) -> tuple:
        """
        Return the stored band grids with the lower limits of their E_z windows.

        Parameters
        ----------
        fermi_level, work_function, electric_field, temperature : float
            Ignored parameters included for subclass compatibility.

        Returns
        -------
        tuple of numpy.ndarray
            The same four arrays as :meth:`Semiconductor.generate_band_structure`.

        Raises
        ------
        ValueError
            If a grid is not a finite, strictly increasing 1-D array, or lies
            outside its band.

        Examples
        --------
        >>> custom = CustomSemiconductor(np.linspace(13.62, 15.0, 139), np.linspace(11.0, 12.5, 151))
        >>> ec1, ec2, ev1, ev2 = custom.generate_band_structure()
        """
        conduction = np.asarray(self.custom_conduction_array, dtype=float)
        valence = np.asarray(self.custom_valence_array, dtype=float)
        for name, grid in (("conduction", conduction), ("valence", valence)):
            if grid.ndim != 1 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
                raise ValueError(f"The {name} band energies must be a finite, strictly "
                                 f"increasing 1-D array.")
        bottom_conduction = self.top_valence + self.band_gap
        # A tolerance of 1e-9 eV, so a grid built from the same band parameters
        # is not rejected over the last bits of its first or last point.
        if conduction.size and conduction[0] < bottom_conduction - 1e-9:
            raise ValueError(f"Conduction band energies start at {conduction[0]:.6g} eV, below "
                             f"the band edge E_C = {bottom_conduction:.6g} eV, where there are "
                             f"no conduction states.")
        if valence.size and valence[-1] > self.top_valence + 1e-9:
            raise ValueError(f"Valence band energies reach {valence[-1]:.6g} eV, above the band "
                             f"top E_V = {self.top_valence:.6g} eV, where there are no valence "
                             f"states.")
        return _get_window_limits(conduction, valence, self.top_valence, self.band_gap,
                                  self.electron_eff_mass, self.hole_eff_mass)

#: How far above the higher of the barrier maximum and the Fermi level a
#: tabulated table has to reach, in units of ``k_B T``, before
#: :class:`DensityOfStatesMetal` stops warning. A table ends where the
#: calculation that produced it ran out of bands, and its density of states
#: falls to zero there for that reason rather than a physical one, so emission
#: from the top of a table is not to be trusted. Measured on a tabulated d band
#: metal, as the share of the current density coming from the table's top 1 eV:
#: at 8 ``k_B T`` of headroom that share is 5e-6, at 10.4 it is 1.6e-4, and it
#: climbs to 3e-3 at 7.6, 0.25 at 4.1 and 0.31 at 1.1. Eight separates the
#: conditions whose answer is set by real states from those whose answer is set
#: by where the band count stopped.
_TABULATED_GRID_MARGIN = 8.0


def _read_tabulated_columns(path, columns):
    """
    Two numeric columns of a whitespace-separated text file, in eV and states/eV.

    Lines that do not parse as numbers are skipped, so a header line -- which
    density of states files usually carry, and rarely mark with a comment
    character -- needs no ``skiprows`` count. The encoding is given explicitly:
    reading without one has been a Windows-only failure here before.
    """
    first, second = columns
    energies, values = [], []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            try:
                numbers = [float(value) for value in fields]
            except ValueError:
                continue
            if max(first, second) >= len(numbers):
                raise ValueError(
                    f"{path} has {len(numbers)} columns, too few for "
                    f"columns={columns}.")
            energies.append(numbers[first])
            values.append(numbers[second])
    if not energies:
        raise ValueError(f"{path} contains no numeric rows.")
    return np.array(energies), np.array(values)


class DensityOfStatesMetal(BandStructure):
    """
    A metal whose energy grid and state count both come from a tabulated density of states.

    The other metal band structures assume a free electron gas, whose density of
    states goes as ``sqrt(E)`` above the bottom of the conduction band. This one
    takes ``g(E)`` from a table -- a DFT calculation, say -- and weights every
    total energy by how many states it actually holds relative to that free
    electron reference,

        R(E) = [g(E) / sqrt(E)] / [g(E_F) / sqrt(E_F)] ,

    normalised so that ``R(E_F) = 1``. The emitter multiplies the total energy
    distribution by ``R`` and rebuilds the normal energy distribution from the
    same double integral, so the current density, the Nottingham heat and both
    distributions stay consistent with each other.

    The energies are the table's own, so the table sets both the range and the
    resolution of the grid. Energies at or below zero are dropped: zero is the
    bottom of the conduction band and the reference for every energy in this
    package, and ``1 / sqrt(E)`` has no meaning below it.

    Parameters
    ----------
    dos_energies : array_like
        Energies of the tabulated density of states, in eV, on the same scale as
        ``fermi_level`` -- that is, measured from the bottom of the conduction
        band, not from the Fermi level. Must be finite and strictly increasing.
    dos_values : array_like
        The density of states at those energies. Any units: only the shape
        survives the normalisation, which is why absolute currents remain
        comparable to the free electron result even for a projected density of
        states with no absolute normalisation of its own.
    fermi_level : float
        The Fermi level, in eV, measured from the bottom of the conduction band.
        Must lie inside the tabulated range, where the table must have states.
    work_function : float
        The work function, in eV. Used only to check that the grid reaches over
        the top of the barrier; the barrier itself holds its own copy.

    Attributes
    ----------
    dos_energies, dos_values : numpy.ndarray
        The table, cropped to positive energies.
    fermi_level, work_function : float
        As above.
    energy_resolution : float
        The table's own step, taken as the median spacing of its energies.

    Notes
    -----
    **What the weight is, and what it is not.** In the supply integral the group
    velocity cancels the Jacobian, ``v_z dk_z = dE / hbar``, so the quantity that
    belongs here is the number of forward-propagating channels at energy E --
    the area of the constant energy surface projected on the surface plane,
    counted with multiplicity -- and not the density of states, which weights
    states by one over the same velocity. Writing ``R = g / sqrt(E)`` assumes the
    two are related as they are for free electrons. Where flat bands dominate
    ``g`` they have small normal velocity and small projected area, so their
    contribution is overestimated; for a d band metal that is a real effect, not
    a rounding one. Treat this as a model of the band structure's influence on
    emission rather than as a first-principles calculation of it. If the number
    of channels ``N(E)`` is known instead, it belongs in the same slot against
    its own free electron reference, which goes as ``E`` and not as ``sqrt(E)``:
    :meth:`get_state_weights` is the only place the ratio is formed.

    **Range, and where this stops being physics.** A table ends where the
    calculation that produced it ran out of bands, and its density of states
    falls to zero at that edge for that reason and not a physical one. Emission
    from the top of a table is therefore an artefact of the band count: the model
    reports no states where a real metal has a free electron-like continuum, and
    the current comes out too low.

    Field emission is untouched by this, because it draws from within a fraction
    of an electronvolt of the Fermi level -- on a tabulated d band metal the
    share of the current density coming from the table's top 1 eV is around
    1e-30 at 300 K, at every field from 1.5 to 7 V/nm. Thermionic emission is
    another matter: it draws from the top of the barrier, which sits close to the
    end of the table, and the same share reaches 0.99 at 1.5 V/nm and 2000 K,
    where the tabulated current density comes out 23 times below the free
    electron one for no physical reason.

    So ``generate_band_structure`` warns when the top of the table clears the
    higher of the barrier maximum and the Fermi level by less than ``8 k_B T``.
    The barrier maximum is the Schottky-Nordheim one,
    ``E_F + phi - 2 sqrt(k_e F / 4)``, as in :class:`SmartMetal`. Take the
    warning seriously: what it flags is not a discretisation error but a result
    set by where the band structure calculation stopped.

    Examples
    --------
    >>> band = DensityOfStatesMetal.from_file(          # doctest: +SKIP
    ...     "dos.txt", fermi_level=10.268, work_function=4.67)
    >>> energies = band.generate_band_structure()       # doctest: +SKIP
    >>> weights = band.get_state_weights(energies)      # doctest: +SKIP
    """

    def __init__(self, dos_energies: np.ndarray, dos_values: np.ndarray,
                 fermi_level: float, work_function: float):
        """
        Initialize the DensityOfStatesMetal instance from a tabulated density of states.
        """
        energies = np.asarray(dos_energies, dtype=float)
        values = np.asarray(dos_values, dtype=float)
        if energies.ndim != 1 or values.ndim != 1:
            raise ValueError("The density of states energies and values must be "
                             "1-D arrays.")
        if energies.size != values.size:
            raise ValueError(f"The density of states has {energies.size} energies "
                             f"but {values.size} values.")
        if not (np.all(np.isfinite(energies)) and np.all(np.isfinite(values))):
            raise ValueError("The density of states must be finite everywhere.")
        if energies.size > 1 and np.any(np.diff(energies) <= 0):
            raise ValueError("The density of states energies must be strictly "
                             "increasing.")

        keep = energies > 0.0
        energies, values = energies[keep], values[keep]
        if energies.size < 2:
            raise ValueError(
                "Fewer than two tabulated energies are positive. Energies are "
                "measured from the bottom of the conduction band, so a table "
                "referenced to the Fermi level has to be shifted by the Fermi "
                "level before it is passed here.")

        # A smearing scheme can put a small negative value in a DFT density of
        # states. Negative states are unphysical and a negative weight would make
        # the distributions negative, so they are removed -- but not quietly.
        negative = values < 0.0
        if negative.any():
            warnings.warn(
                f"{int(negative.sum())} of {values.size} tabulated density of "
                f"states values are negative (most negative "
                f"{values.min():.3g}); they have been set to zero.",
                RuntimeWarning, stacklevel=2)
            values = np.maximum(values, 0.0)

        super().__init__(energy_resolution=float(np.median(np.diff(energies))))
        self.dos_energies = energies
        self.dos_values = values
        self.fermi_level = float(fermi_level)
        self.work_function = float(work_function)
        self._weight_cache = None

        if not energies[0] < self.fermi_level < energies[-1]:
            raise ValueError(
                f"The Fermi level {self.fermi_level:.6g} eV is outside the "
                f"tabulated range {energies[0]:.6g} to {energies[-1]:.6g} eV. "
                f"Both are measured from the bottom of the conduction band; a "
                f"table whose energies are referenced to the Fermi level has to "
                f"be shifted before it is passed here.")
        if np.interp(self.fermi_level, energies, values) <= 0.0:
            raise ValueError(
                f"The tabulated density of states is zero at the Fermi level "
                f"{self.fermi_level:.6g} eV, so it describes no metal and the "
                f"weights cannot be normalised there.")

    @classmethod
    def from_file(cls, path, fermi_level: float, work_function: float,
                  columns=(0, 1)) -> "DensityOfStatesMetal":
        """
        Build the band structure from a two-column text file.

        Parameters
        ----------
        path : str or pathlib.Path
            A whitespace-separated text file. Header lines and ``#`` comments are
            skipped, so no row count is needed.
        fermi_level, work_function : float
            As for the constructor, in eV.
        columns : tuple of int, default (0, 1)
            Which columns hold the energies and the density of states. Files with
            several projections in them are common, so the pair is explicit.

        Returns
        -------
        DensityOfStatesMetal

        Examples
        --------
        >>> band = DensityOfStatesMetal.from_file(      # doctest: +SKIP
        ...     "examples/dos.txt", fermi_level=10.268, work_function=4.67)
        """
        energies, values = _read_tabulated_columns(path, columns)
        return cls(energies, values, fermi_level=fermi_level,
                   work_function=work_function)

    def get_state_weights(self, energy_array: np.ndarray) -> np.ndarray:
        """
        The ratio ``R(E)`` of tabulated to free electron states, normalised to one at E_F.

        Zero outside the tabulated range, where the table says there are no
        states. Evaluated by interpolation, which is exact on the grid
        :meth:`generate_band_structure` returns, since that is the table's own.

        Parameters
        ----------
        energy_array : numpy.ndarray
            Total energies, in eV.

        Returns
        -------
        numpy.ndarray
        """
        if self._weight_cache is None or self._weight_cache[0] != self.fermi_level:
            # Normalise the ratio itself at the Fermi level, rather than forming
            # the reference from g and sqrt(E) separately: the Fermi level
            # rarely lands on a tabulated energy, and interpolating the two
            # factors apart leaves R(E_F) off by the interpolation error of
            # sqrt rather than by round-off. A table that is already the free
            # electron one then gives R == 1 exactly, and takes the same path
            # as a band structure that has no table at all.
            ratio = self.dos_values / np.sqrt(self.dos_energies)
            reference = np.interp(self.fermi_level, self.dos_energies, ratio)
            self._weight_cache = (self.fermi_level, ratio / reference)
        return np.interp(np.asarray(energy_array, dtype=float), self.dos_energies,
                         self._weight_cache[1], left=0.0, right=0.0)

    def generate_band_structure(self, fermi_level: float = None,
                                work_function: float = None,
                                electric_field: float = 3.0,
                                temperature: float = 300.) -> np.ndarray:
        """
        The tabulated energies, which set both the range and the resolution of the grid.

        Parameters
        ----------
        fermi_level : float, optional
            The Fermi level, in eV. Defaults to the instance's own.
        work_function : float, optional
            The work function, in eV. Defaults to the instance's own.
        electric_field : float, default 3.0
            The field, in V/nm. Used only for the range check.
        temperature : float, default 300.0
            The temperature, in K. Used only for the range check.

        Returns
        -------
        numpy.ndarray
            The tabulated energies, cropped to positive values.

        Warns
        -----
        RuntimeWarning
            If the table ends too close to the top of the barrier or to the
            Fermi level for the emission to be complete. See the class notes.

        Examples
        --------
        >>> band = DensityOfStatesMetal(           # doctest: +SKIP
        ...     energies, dos, fermi_level=10.268, work_function=4.67)
        >>> band.generate_band_structure(electric_field=5.0, temperature=300.0)
        """
        fermi = self.fermi_level if fermi_level is None else fermi_level
        work = self.work_function if work_function is None else work_function
        self._warn_if_truncated(fermi, work, electric_field, temperature)
        return self.dos_energies

    def _warn_if_truncated(self, fermi_level, work_function, electric_field,
                           temperature):
        """
        Warn when the table ends before the emission does.

        Two ways to lose current, both silent: the table can stop below the top
        of the barrier, which removes the whole over-barrier channel, and it can
        stop before the Fermi tail has died away. The higher of the two energies
        is what the grid has to clear.
        """
        barrier_top = fermi_level + work_function
        if electric_field > 0:
            barrier_top -= 2.0 * np.sqrt(const.IMAGE_PREFACTOR * electric_field)
        needed = (max(barrier_top, fermi_level)
                  + _TABULATED_GRID_MARGIN * const.KB * temperature)
        top = self.dos_energies[-1]
        if top < needed:
            warnings.warn(
                f"The tabulated density of states ends at {top:.4g} eV, below "
                f"the {needed:.4g} eV this grid needs at {electric_field:.4g} "
                f"V/nm and {temperature:.4g} K (the barrier top is "
                f"{barrier_top:.4g} eV). The current density and the Nottingham "
                f"heat will be underestimated. Use a table that reaches higher, "
                f"or restrict the field and temperature to where this one does.",
                RuntimeWarning, stacklevel=3)
