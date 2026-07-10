"""
Potential Barrier Simulation Module.

This module defines abstract and concrete classes to model the spatial profile 
of potential energy barriers at electron-emitting interfaces. It includes standard 
planar models with image-charge corrections (Schottky effect) as well as localized 
curvature models suitable for nano-emitter tips with small radii.

Dependencies
------------
numpy : Array manipulation and vectorized math operations.
abc : Base structure handling for abstract classes.
getelec.constants : Physical constants library containing electrostatic parameters.
"""

import numpy as np
from abc import ABC, abstractmethod
from getelec import constants as const

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
        # Initialise an array to store the potential data
        potential = np.zeros_like(x_points)

        # Define the potential for x > 0
        mask_outside = x_points > 0
        x_outside = x_points[mask_outside]

        # Calculate the potential using the smooth form
        potential[mask_outside] = self.fermi_level + self.work_function - self.electric_field * x_outside - const.COULOMB_CONST / (4 * x_outside)
        # # Inside the metal V is constant and set to 0 V
        # potential[~mask_outside] = 0.0

        # Ensure that in the region near the surface there are no negative energies
        potential[(x_points < 1) & (potential < 0)] = 0.0

        return potential

class SmallRadiiPotential(Barrier):
    """
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
    radius : float, default 5.0
        The physical radius of curvature of the micro/nano-emitter tip.
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
    """

    def __init__(self, fermi_level: float = 9.5, work_function: float = 4.5, electric_field: float = 3.0, radius: float = 5.0, gamma: float = 100):
        """
        Initialize the SmallRadiiPotential boundary model.
        """
        self.fermi_level = fermi_level
        self.work_function = work_function
        self.electric_field = electric_field
        self.radius = radius
        self.gamma = gamma

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
        >>> tip_barrier = SmallRadiiPotential(radius=10.0, gamma=50)
        >>> coordinates = np.array([0.5, 1.0, 5.0])
        >>> tip_barrier.get_potential(coordinates)
        """
        # Initialise an array to store the potential data
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
        potential[(x_points < 1) & (potential < 0)] = 0.0

        return potential