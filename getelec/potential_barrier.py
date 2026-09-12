"""
Potential Barrier Simulation Module.

This module defines abstract and concrete classes to model the spatial profile
of potential energy barriers at electron-emitting interfaces. It includes standard
planar models with image-charge corrections (Schottky effect), localized
curvature models suitable for nano-emitter tips with small radii, and the pure
triangular barrier with no image charge.

Dependencies
------------
numpy : Array manipulation and vectorized math operations.
abc : Base structure handling for abstract classes.
getelec.constants : Physical constants library containing electrostatic parameters.
"""

import inspect
import itertools
import warnings

import numpy as np

#: Range of tip radii (nm) over which SmallRadiiPotential is valid; the class
#: warns outside it. Both bounds are set by the current density. Below 20 nm the
#: current depends on the shape of the emitter beyond its apex, which the
#: barrier's two parameters (radius and gamma) do not describe. Above 1000 nm the
#: tip is planar for the current: the curvature changes it by 2.8%, 1.1% and
#: 0.6% at 3, 5 and 7 V/nm (phi = 4.5 eV, gamma = 100), less further up, and
#: SchottkyPotential gives the same result.
RADIUS_VALIDITY = (20.0, 1000.0)
from abc import ABC, abstractmethod
from getelec import constants as const


def get_image_cutoff(fermi_level, work_function, electric_field):
    """
    Position below which the image term makes the potential unphysical, in nm.

    ``-k_e/(4x)`` diverges at the surface, so the barrier has to be cut off
    somewhere. The right place is where the potential first crosses zero on the
    way up -- the inner root of ``E_F + phi - F x - k_e/(4x) = 0``,

        x_c = (W - sqrt(W^2 - k_e F)) / (2F),   W = E_F + phi .

    Below it the expression is negative only because of the divergence; above
    it, any negative value is the real barrier falling away past its outer
    turning point and must be kept.

    Parameters
    ----------
    fermi_level, work_function, electric_field : float
        eV, eV, V/nm.

    Returns
    -------
    float
    """
    total = fermi_level + work_function
    discriminant = total ** 2 - const.COULOMB_CONST * electric_field
    if discriminant <= 0:
        # The field has suppressed the barrier entirely; there is no crossing,
        # so fall back to the position of what is left of the maximum.
        return float(np.sqrt(const.COULOMB_CONST / (4.0 * electric_field)))
    return float((total - np.sqrt(discriminant)) / (2.0 * electric_field))


class Barrier(ABC):
    """
    Abstract base class for potential barriers.

    Defines the shared structural interface for computing spatial potential energy 
    profiles relative to an emitting surface boundary.
    """

    @abstractmethod
    def get_potential(self, x_points: np.ndarray) -> np.ndarray:
        """
        Calculate the potential energy profile along spatial coordinates.

        Parameters
        ----------
        x_points : numpy.ndarray
            1D array containing spatial distance coordinates relative to the surface (typically in nm).

        Returns
        -------
        potential : numpy.ndarray
            1D array containing the corresponding potential energy values (typically in eV).
        """
        pass


def get_barrier_parameters(potential):
    """
    Every parameter a barrier's shape depends on, as a hashable tuple.

    Read from the instance's own attributes rather than a fixed list, so a
    subclass that introduces a new parameter is distinguished automatically.
    A fixed list silently collided: two barriers differing only in a new
    attribute produced the same cache key, and the solver returned the first
    one's potential for both.

    Barriers that compute their profile from something other than plain
    attributes should override ``parameters`` to include it.
    """
    if hasattr(potential, "parameters"):
        values = potential.parameters()
    else:
        values = {name: value for name, value in vars(potential).items()
                  if not name.startswith("_")}
    out = [type(potential).__name__]
    for name in sorted(values):
        value = values[name]
        if isinstance(value, np.ndarray):
            out.append((name, value.shape, value.tobytes()))
        elif isinstance(value, (int, float, str, bool, type(None))):
            out.append((name, value))
        else:
            out.append((name, repr(value)))
    return tuple(out)


class SchottkyPotential(Barrier):
    """
    Planar surface potential model with Schottky image-charge correction.

    Models the classical combination of a material's inherent work function barrier, 
    a uniform linear applied external electric field, and the image-charge reduction 
    term felt by a departing electron outside a planar conductive surface.

    Parameters
    ----------
    fermi_level : float, default 9.5
        The interior Fermi energy level of the material system.
    work_function : float, default 4.5
        The inherent work function barrier of the surface.
    electric_field : float, default 3.0
        The applied external electric field magnitude acting on the interface.

    Attributes
    ----------
    fermi_level : float
        Stored value for the material chemical potential.
    work_function : float
        Stored value for the inherent surface work function.
    electric_field : float
        Stored value for the active external electric field.
    """

    def __init__(self, fermi_level: float = 9.5, work_function: float = 4.5, electric_field: float = 3.0):
        """
        Initialize the SchottkyPotential boundary model.
        """
        self.fermi_level = fermi_level
        self.work_function = work_function
        self.electric_field = electric_field

    def get_potential(self, x_points: np.ndarray) -> np.ndarray:
        """
        Evaluate the planar Schottky potential profile across a spatial coordinate array.

        Computes the classical image-charge curve for positions outside the emitter 
        and protects against potential energy divergences near the raw boundary layer 
        by floor-clamping negative artifact outputs near the surface.

        Parameters
        ----------
        x_points : numpy.ndarray
            1D array containing spatial distance coordinates where potential is to be evaluated.

        Returns
        -------
        potential : numpy.ndarray
            1D array containing computed potential energy values across the input domain.

        Examples
        --------
        >>> barrier = SchottkyPotential(fermi_level=9.5, work_function=4.5, electric_field=3.0)
        >>> distances = np.array([-0.5, 0.0, 1.0, 2.0])
        >>> barrier.get_potential(distances)
        """
        # float dtype is forced: np.zeros_like on an integer grid would give an
        # integer array and silently truncate the potential values written into it.
        x_points = np.asarray(x_points, dtype=float)
        potential = np.zeros_like(x_points)

        # Define the potential for x > 0
        mask_outside = x_points > 0
        x_outside = x_points[mask_outside]

        # Calculate the potential using the smooth form
        potential[mask_outside] = self.fermi_level + self.work_function - self.electric_field * x_outside - const.COULOMB_CONST / (4 * x_outside)
        # # Inside the metal V is constant and set to 0 V
        # potential[~mask_outside] = 0.0

        # Ensure that in the region near the surface there are no negative energies
        # Clamp only where the image term diverges, not at a fixed 1 nm.
        cutoff = get_image_cutoff(self.fermi_level, self.work_function,
                              self.electric_field)
        potential[x_points < cutoff] = 0.0

        return potential

class SmallRadiiPotential(Barrier):
    r"""
    Curved surface potential model for localized nano-emitter geometries.

    Accounts for field enhancement and modified image-force interactions experienced 
    at the boundary tips of high-curvature nanostructures (such as field emission needles), 
    incorporating a tip radius and geometric scaling parameter ($\gamma$).

    Parameters
    ----------
    fermi_level : float, default 9.5
        The interior Fermi energy level of the material system.
    work_function : float, default 4.5
        The inherent work function barrier of the surface.
    electric_field : float, default 3.0
        The macroscopic baseline applied external electric field.
    radius : float, default 20.0
        The physical radius of curvature of the micro/nano-emitter tip, in nm.
        The barrier is valid for 20-1000 nm (:data:`RADIUS_VALIDITY`); outside
        that range the class warns.
    gamma : float, default 100
        Geometric field enhancement/correction multiplier coefficient.

    Attributes
    ----------
    fermi_level : float
        Stored value for the material chemical potential.
    work_function : float
        Stored value for the inherent surface work function.
    electric_field : float
        Stored value for the active baseline electric field.
    radius : float
        Stored value for the apex radius of curvature.
    gamma : float
        Stored value for the geometric enhancement tracking constant.

    Notes
    -----
    Across its valid range the potential is close to the planar
    Schottky-Nordheim barrier of :class:`SchottkyPotential`: at R = 20 nm it
    lies within 0.3 eV of it across the barrier at the Fermi level (under 7% of
    the work function) at 3-8 V/nm, for gamma = 100 and E_F + phi = 12 eV. The
    current depends exponentially on the barrier, so it is not close: at
    R = 20 nm it is 0.22, 0.57 and 0.79 times the planar current density at 3,
    5 and 8 V/nm, and at 50 nm 0.56, 0.80 and 0.91 times. The difference falls
    as 1/R; at 1000 nm, the top of the range, it is 2.8%, 1.1% and 0.6% at 3,
    5 and 7 V/nm.

    On the axis the applied term is exact to second order in x/R: its curvature
    is the one Laplace's equation requires at an apex of radius R. The
    third-order term, c3 F x^3 / R^2, is set by the shape of the emitter away
    from its apex, which radius and gamma do not describe: c3 = 4/3 for an
    ellipsoidal tip of any aspect ratio, about 1.05 for a hemisphere on a
    cylindrical post at gamma = 100, and gamma / (gamma - 1) here. The current
    density is sensitive to it, most of all near the outer turning point.
    Against the exact electrostatics of the two shapes (gamma = 100,
    phi = 4.5 eV, 300 K) this barrier under-predicts the current density at
    R = 20 nm by 3.3%, 0.7% and 0.26% at 3, 5 and 7 V/nm for an ellipsoidal
    tip, and by at most 0.5% for a hemisphere on a post; at R = 10 nm the
    ellipsoidal figures are 16%, 3.1% and 1.1%. The difference grows roughly as
    phi^(7/2) / (F^3 R^2), which is why the class warns below 20 nm.
    """

    def __init__(self, fermi_level: float = 9.5, work_function: float = 4.5, electric_field: float = 3.0, radius: float = 20.0, gamma: float = 100):
        """
        Initialize the SmallRadiiPotential boundary model.
        """
        self.fermi_level = fermi_level
        self.work_function = work_function
        self.electric_field = electric_field
        self.radius = radius
        self.gamma = gamma
        if radius < RADIUS_VALIDITY[0]:
            warnings.warn(
                f"radius = {radius} nm is below the range "
                f"{RADIUS_VALIDITY[0]}-{RADIUS_VALIDITY[1]} nm over which this "
                f"curvature-corrected barrier is valid. Below it the current "
                f"density depends on the shape of the emitter beyond its apex, "
                f"which radius and gamma do not describe: for an ellipsoidal "
                f"tip it is under-predicted by about 16% at 10 nm and 3 V/nm "
                f"(phi = 4.5 eV), and more at smaller radii or lower fields.",
                UserWarning, stacklevel=2)
        elif radius > RADIUS_VALIDITY[1]:
            warnings.warn(
                f"radius = {radius} nm is above the range "
                f"{RADIUS_VALIDITY[0]}-{RADIUS_VALIDITY[1]} nm of this "
                f"curvature-corrected barrier. There the tip is effectively "
                f"planar: the curvature changes the current density by about 1% "
                f"or less at 5 V/nm and above (phi = 4.5 eV), so "
                f"SchottkyPotential gives the same result.",
                UserWarning, stacklevel=2)

    def get_potential(self, x_points: np.ndarray) -> np.ndarray:
        """
        Evaluate the non-planar curvature-corrected potential profile across spatial points.

        Modifies standard linear field drops and planar image laws into hyperbolic 
        or spherically-weighted components to faithfully map emission barriers 
        near nanoscale apex points.

        Parameters
        ----------
        x_points : numpy.ndarray
            1D array containing spatial distance coordinates where potential is to be evaluated.

        Returns
        -------
        potential : numpy.ndarray
            1D array containing computed potential energy values across the input domain.

        Examples
        --------
        >>> tip_barrier = SmallRadiiPotential(radius=20.0, gamma=50)
        >>> coordinates = np.array([0.5, 1.0, 5.0])
        >>> tip_barrier.get_potential(coordinates)
        """
        # float dtype is forced: np.zeros_like on an integer grid would give an
        # integer array and silently truncate the potential values written into it.
        x_points = np.asarray(x_points, dtype=float)
        potential = np.zeros_like(x_points)

        # Define the potential for x > 0
        mask_outside = x_points > 0
        x_outside = x_points[mask_outside]

        # Calculate the potential using the smooth form
        term1 = self.fermi_level + self.work_function
        term2 = self.electric_field * (((self.radius*(self.gamma-1)*x_outside)+x_outside**2)/(self.gamma*x_outside+self.radius*(self.gamma-1)))
        term3 = const.COULOMB_CONST / (4 * x_outside * (1 + (x_outside/(2*self.radius))))
        potential[mask_outside] = term1 - term2 - term3
        # # Inside the metal V is constant and set to 0 V
        # potential[~mask_outside] = 0.0

        # Ensure that in the region near the surface there are no negative energies
        # Clamp only where the image term diverges, not at a fixed 1 nm.
        cutoff = get_image_cutoff(self.fermi_level, self.work_function,
                              self.electric_field)
        potential[x_points < cutoff] = 0.0

        return potential

class TriangularPotential(Barrier):
    """
    Planar triangular barrier: the vacuum level tilted by the field, with no image charge.

    The barrier of Fowler and Nordheim (1928),

        V(x) = E_F + phi - F x    for x > 0,     V(x) = 0 inside the metal.

    Without the image term there is no Schottky lowering: the maximum is the
    full ``E_F + phi``, at the surface, and the barrier is taller and wider than
    :class:`SchottkyPotential` at the same field, so it emits less. Past the
    outer turning point the potential keeps falling below zero, as the applied
    field makes it; nothing is clamped.

    Parameters
    ----------
    fermi_level : float, default 9.5
        The interior Fermi energy level of the material system, in eV.
    work_function : float, default 4.5
        The inherent work function barrier of the surface, in eV.
    electric_field : float, default 3.0
        The applied external electric field, in V/nm.

    Attributes
    ----------
    fermi_level : float
        Stored value for the material chemical potential.
    work_function : float
        Stored value for the inherent surface work function.
    electric_field : float
        Stored value for the active external electric field.

    Notes
    -----
    This is the one barrier with a closed-form transmission:
    :class:`~getelec.transmission_solutions.AiryTriangular` (``method="airy"``)
    is exact for it, below and above the barrier top. The analytic
    :class:`~getelec.transmission_solutions.WKB` solver is the
    Schottky-Nordheim form and does not describe this barrier.

    The potential jumps by ``E_F + phi`` at the surface, which a grid solver
    resolves only to first order in its step -- by an amount set by where the
    surface falls between two grid points, so a node landing on the surface
    costs 1.3% in current density at ``h = 1e-3`` nm. The Noumerov grid puts
    the surface halfway between two nodes, the placement that cancels the
    first-order term: against the exact solution the current density is within
    1e-5 at the default step and the transmission within 6e-5, measured for
    ``E_F + phi`` = 5 to 13.5 eV, 1-12 V/nm and 300-2000 K.
    """

    def __init__(self, fermi_level: float = 9.5, work_function: float = 4.5, electric_field: float = 3.0):
        """
        Initialize the TriangularPotential boundary model.
        """
        self.fermi_level = fermi_level
        self.work_function = work_function
        self.electric_field = electric_field

    def get_potential(self, x_points: np.ndarray) -> np.ndarray:
        """
        Evaluate the triangular potential profile across a spatial coordinate array.

        Parameters
        ----------
        x_points : numpy.ndarray
            1D array containing spatial distance coordinates where potential is to be evaluated, in nm.

        Returns
        -------
        potential : numpy.ndarray
            1D array containing computed potential energy values across the input domain, in eV.

        Examples
        --------
        >>> barrier = TriangularPotential(fermi_level=7.5, work_function=4.5, electric_field=5.0)
        >>> barrier.get_potential(np.array([-0.5, 0.0, 1.0, 3.0]))
        array([ 0.,  0.,  7., -3.])
        """
        # float dtype is forced: np.zeros_like on an integer grid would give an
        # integer array and silently truncate the potential values written into it.
        x_points = np.asarray(x_points, dtype=float)
        potential = np.zeros_like(x_points)

        # Inside the metal V is 0; outside, the vacuum level falls linearly.
        mask_outside = x_points > 0
        potential[mask_outside] = (self.fermi_level + self.work_function
                                   - self.electric_field * x_points[mask_outside])
        return potential


#: Distinguishes one Customised instance from another in the solver's potential
#: cache. The barrier's own parameters cannot do it alone: a callable enters the
#: cache key by its repr, which is its memory address, and an address freed by
#: one function can be reused by the next.
_CUSTOM_SERIAL = itertools.count()

#: The parameters an emitter sets on every component at once. A Customised
#: barrier carries them so that a potential depending on any of them cannot
#: silently disagree with the supply function or the band structure.
_SYNCED_PARAMETERS = ("fermi_level", "work_function", "electric_field",
                      "temperature")


def get_accepted_keywords(function):
    """
    The keyword names a callable will accept, past its first argument.

    Returns ``"any"`` for a callable taking ``**kwargs``, ``None`` when the
    signature cannot be read (a C callable), and otherwise the set of names.
    The first positional parameter is left out: it is the position array.
    """
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return None
    names = set()
    for index, (name, parameter) in enumerate(signature.parameters.items()):
        if parameter.kind is parameter.VAR_KEYWORD:
            return "any"
        if parameter.kind is parameter.VAR_POSITIONAL:
            continue
        if index == 0:
            continue
        names.add(name)
    return names


class Customised(Barrier):
    r"""
    Barrier built from a potential the user supplies.

    Takes either a callable ``V(x)`` or a tabulated profile and presents it to
    the solvers exactly as the shipped barriers do, so a potential worked out
    elsewhere -- a screened or self-consistent one, a profile from another code,
    measured points -- can be run through the same transmission solvers,
    emitters and distributions without subclassing anything.

    Parameters
    ----------
    potential : callable or array_like
        Either ``V(x)``, taking positions in nm and returning energies in eV
        pointwise (same shape in, same shape out), or an array of potential
        values to be interpolated, in which case ``positions`` is required.
    positions : array_like, optional
        Sample positions for a tabulated potential, nm, strictly increasing.
        Not accepted for a callable, which is evaluated wherever the solver
        asks.
    fermi_level, work_function, electric_field, temperature : float, optional
        eV, eV, V/nm and K. Not used to build the potential -- that is entirely
        the user's -- but passed to the callable if it takes parameters of
        those names, and the first three are what
        :class:`~getelec.transmission_solver.Noumerov` reads to size its
        integration domain (``auto_domain``). Without them the domain falls
        back to the solver's ``max_barrier_width``, 3 nm by default, which is
        too short for a low field. These four are the parameters
        ``update_params`` sets on every component at once, so handing this
        barrier to :func:`~getelec.metal_emitter` replaces them with that
        call's arguments -- which is what keeps a barrier that depends on the
        temperature from disagreeing with the supply function about it.
        :func:`~getelec.semiconductor_emitter` sets none of them: give them
        here.
    zero_inside : bool, default True
        Hold the potential at 0 for ``x <= 0`` and evaluate the user's
        potential only outside, which is the convention of every other barrier
        here: the energy zero is the bottom of the conduction band, the metal
        is flat, and the vacuum level at the surface is ``E_F + phi``. Set it
        to False only if the potential describes the metal side itself; the
        solver still needs it flat over the last two nodes of the metal, where
        the plane-wave match is made.
    **extra_parameters
        Any further named parameters of the callable, kept on the barrier and
        passed at every evaluation. A name the callable will not accept is an
        error, not a value silently dropped.

    Attributes
    ----------
    potential_function : callable or None
        The user's callable, or None for a tabulated potential.
    table_positions, table_potential : numpy.ndarray or None
        The tabulated profile, or None for a callable.
    fermi_level, work_function, electric_field, temperature : float or None
        Stored values, passed to the callable and used to size the domain.
    zero_inside : bool
        Whether the metal side is held at zero.
    extra_parameters : dict
        The further parameters passed to the callable.

    Notes
    -----
    Units are this package's throughout: x in nm, V in eV, measured from the
    same zero as the other barriers (0 inside the metal).

    The potential must fall below the emitted energies before the end of the
    solver's grid. A barrier still above them at the last node is not solved
    slightly wrongly, it is meaningless -- that is what ``auto_domain``
    protects against for the shipped barriers, and it needs the three
    parameters above.

    Nothing is clamped near the surface. A potential that diverges there -- an
    unscreened image term -- has to be cut off by the caller;
    :func:`get_image_cutoff` is the position the shipped barriers use.

    Only the grid solvers read the barrier. ``method="wkb"`` and
    ``method="airy"`` build their transmission from E_F, phi and F alone, so
    with a barrier of your own they return the Schottky-Nordheim and triangular
    answers, not yours. Use ``method="noumerov"``, the default.

    The solvers cache the profile against the barrier's parameters, so a
    callable is evaluated once per grid rather than once per energy. Its result
    must therefore depend only on its arguments: a callable reading a global
    that changes between calls returns the first profile again.

    Examples
    --------
    >>> triangular = Customised(
    ...     lambda x, fermi_level, work_function, electric_field:
    ...     fermi_level + work_function - electric_field * x,
    ...     fermi_level=7.5, work_function=4.5, electric_field=5.0)
    >>> triangular.get_potential(np.array([-0.5, 1.0, 3.0]))
    array([ 0.,  7., -3.])
    """

    def __init__(self, potential, positions=None, fermi_level: float = None,
                 work_function: float = None, electric_field: float = None,
                 temperature: float = None, zero_inside: bool = True,
                 **extra_parameters):
        """
        Initialize the Customised boundary model.
        """
        self._serial = next(_CUSTOM_SERIAL)
        self._signature_source = None
        self._accepted_keywords = None

        if callable(potential):
            if positions is not None:
                raise ValueError(
                    "positions is for a tabulated potential. A callable is "
                    "evaluated at whatever positions the solver asks for, so "
                    "there is nothing to tabulate it on.")
            self.potential_function = potential
            self.table_positions = None
            self.table_potential = None
        else:
            values = np.asarray(potential, dtype=float)
            if positions is None:
                raise ValueError(
                    "A tabulated potential needs the positions it was sampled "
                    "at: Customised(values, positions). Pass a callable V(x) "
                    "instead to have it evaluated on the solver's own grid.")
            grid = np.asarray(positions, dtype=float)
            if grid.ndim != 1 or grid.shape != values.shape:
                raise ValueError(
                    f"positions {grid.shape} and the potential {values.shape} "
                    f"must be one-dimensional arrays of the same length.")
            if grid.size < 2 or np.any(np.diff(grid) <= 0):
                raise ValueError(
                    "positions must be strictly increasing, with at least two "
                    "points; np.interp gives silently wrong values otherwise.")
            self.potential_function = None
            self.table_positions = grid
            self.table_potential = values

        self.fermi_level = fermi_level
        self.work_function = work_function
        self.electric_field = electric_field
        self.temperature = temperature
        self.zero_inside = bool(zero_inside)
        self.extra_parameters = dict(extra_parameters)

        if self.potential_function is not None:
            accepted = self._get_accepted_keywords()
            if isinstance(accepted, set):
                unused = sorted(set(self.extra_parameters) - accepted)
                if unused:
                    name = getattr(potential, "__name__", repr(potential))
                    raise ValueError(
                        f"{unused} are not parameters of {name}, so they would "
                        f"be kept on the barrier and never reach it. Its "
                        f"parameters are {sorted(accepted)}.")
        elif self.extra_parameters:
            raise ValueError(
                f"{sorted(self.extra_parameters)} were given with a tabulated "
                f"potential, which takes no parameters. Pass a callable to "
                f"have parameters passed to it.")

    def parameters(self):
        """
        What the profile depends on, for :func:`get_barrier_parameters`.

        The extra parameters are flattened out of their dict so that an array
        among them enters the cache key by its contents, as it would if it were
        an attribute; a dict enters by ``repr``, which abbreviates a long array
        to ``...`` and so collides. The serial number keeps two instances
        distinct whatever they hold.
        """
        values = {name: value for name, value in vars(self).items()
                  if not name.startswith("_") and name != "extra_parameters"}
        for name, value in self.extra_parameters.items():
            values["parameter:" + name] = value
        values["serial"] = self._serial
        return values

    def _get_accepted_keywords(self):
        """The callable's keyword names, re-read if the callable is replaced."""
        if self._signature_source is not self.potential_function:
            self._signature_source = self.potential_function
            self._accepted_keywords = get_accepted_keywords(self.potential_function)
        return self._accepted_keywords

    def _get_call_parameters(self):
        """The stored parameters the callable will take, as keyword arguments."""
        stored = dict(self.extra_parameters)
        for name in _SYNCED_PARAMETERS:
            value = getattr(self, name)
            if value is not None:
                stored[name] = value
        accepted = self._get_accepted_keywords()
        if accepted is None:
            return {}
        if accepted == "any":
            return stored
        return {name: value for name, value in stored.items() if name in accepted}

    def _evaluate(self, x_points: np.ndarray) -> np.ndarray:
        """The user's potential at these positions, with its shape checked."""
        if self.potential_function is not None:
            values = np.asarray(
                self.potential_function(x_points, **self._get_call_parameters()),
                dtype=float)
            if values.shape != x_points.shape:
                raise ValueError(
                    f"the potential returned shape {values.shape} for "
                    f"{x_points.shape} positions. It has to be evaluated "
                    f"pointwise: one energy per position, same shape out as in.")
            return values

        low, high = self.table_positions[0], self.table_positions[-1]
        if x_points.size and (x_points.min() < low or x_points.max() > high):
            warnings.warn(
                f"positions from {x_points.min():.4g} to {x_points.max():.4g} nm "
                f"were asked for, outside the tabulated {low:.4g} to {high:.4g} "
                f"nm. Beyond the table the end values are held constant, which "
                f"flattens the barrier where it should go on falling, and a "
                f"barrier that never falls below the emitted energies gives a "
                f"meaningless transmission. Tabulate further out, or shorten "
                f"the solver's domain (x_metal, x_vac_plus, max_barrier_width).",
                UserWarning, stacklevel=4)
        return np.interp(x_points, self.table_positions, self.table_potential)

    def get_potential(self, x_points: np.ndarray) -> np.ndarray:
        """
        Evaluate the user's potential across a spatial coordinate array.

        Parameters
        ----------
        x_points : numpy.ndarray
            1D array containing spatial distance coordinates where potential is
            to be evaluated, in nm.

        Returns
        -------
        potential : numpy.ndarray
            1D array containing computed potential energy values across the
            input domain, in eV.

        Raises
        ------
        ValueError
            If the potential does not come back pointwise, or is not finite
            everywhere on the grid -- which a solver would otherwise turn into
            silent NaNs in the transmission.

        Examples
        --------
        >>> tabulated = Customised(np.array([0.0, 5.0, 2.0]),
        ...                        np.array([0.0, 1.0, 2.0]))
        >>> tabulated.get_potential(np.array([-1.0, 0.5, 1.5]))
        array([0. , 2.5, 3.5])
        """
        # float dtype is forced: np.zeros_like on an integer grid would give an
        # integer array and silently truncate the potential values written into it.
        x_points = np.asarray(x_points, dtype=float)

        if self.zero_inside:
            # Evaluated only outside, as the shipped barriers are: a potential
            # that diverges at the surface must not be called at x <= 0 merely
            # to have the result thrown away.
            potential = np.zeros_like(x_points)
            mask_outside = x_points > 0
            if mask_outside.any():
                potential[mask_outside] = self._evaluate(x_points[mask_outside])
        else:
            potential = self._evaluate(x_points)

        if not np.all(np.isfinite(potential)):
            bad = int(np.count_nonzero(~np.isfinite(potential)))
            raise ValueError(
                f"the potential is not finite at {bad} of {potential.size} "
                f"positions. The solver would return NaN transmissions without "
                f"saying why. Cut off any divergence at the surface -- "
                f"get_image_cutoff is where the shipped barriers do it.")
        return potential
