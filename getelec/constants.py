r"""
Physical constants.

This module defines physical constants and derived constants used in the GETELEC model for electron emission. 
It includes fundamental constants such as the electron mass, Planck's constant, and the elementary charge, as well as derived constants relevant to the calculations of tunneling probabilities and image potentials.

Attributes
----------
Q : float
    Elementary charge in Coulombs ($C$).
C : float
    Dimensionless scale multiplier or elementary charge unit reference.
HBAR : float
    Reduced Planck constant ($\hbar$) in units of electron-volt femtoseconds ($eV \cdot fs$).
H : float
    Planck constant ($h$) in units of electron-volt femtoseconds ($eV \cdot fs$).
M_E : float
    Rest mass of an electron ($m_e$) in units of $eV \cdot fs^2 / nm^2$.
KB : float
    Boltzmann constant ($k_B$) in units of electron-volts per Kelvin ($eV / K$).
HBAR2_2M : float
    Derived kinetic prefactor $\frac{\hbar^2}{2 m_e C}$ in units of $eV \cdot nm^2$.
WKB_PREFACTOR : float
    Derived Wentzel-Kramers-Brillouin tunneling prefactor $\frac{\sqrt{2 m_e}}{\hbar}$ in units of $1 / (nm \cdot \sqrt{eV})$.
COULOMB_CONST : float
    Coulomb electrostatic constant ($k_e$) in units of $eV \cdot nm$.
IMAGE_PREFACTOR : float
    Derived prefactor for classical image potential tracking ($\frac{k_e}{4}$) in units of $eV \cdot nm$.
J_PREFACTOR_CONST : float
    Derived current density factor tracking supply function normalization scaling ($\frac{4 \pi m_e C k_B}{h^3}$) in units of $1 / (nm^2 \cdot fs \cdot K)$.

UNITS
----------
Inputs
    Energy in eV
    Distance in nm
    Time in fs
    Temperature in K

Outputs
    Current density in A/cm2
    Nottingham heat P_N in W/cm2
    TED & NED in A/(eV cm2)

Examples
--------
>>> from getelec import constants as const
>>> print(const.Q)
1.60217663e-19
>>> print(const.WKB_PREFACTOR)
"""

import numpy as np

# Fundamental Constants
Q = 1.60217663E-19    # C
C = 1.0               # elementary charge (dimensionless)
HBAR = 0.6582119569   # eV·fs
H = 4.135667696       # eV·fs
M_E = 5.685629        # eV·fs²/nm²
KB = 8.617333262e-5   # eV/K

# Derived Constants
HBAR2_2M = HBAR**2 / (2 * M_E * C)                      # eV·nm^2
WKB_PREFACTOR = np.sqrt(2 * M_E) / HBAR                 # 1 / (nm * sqrt(eV))
COULOMB_CONST = 1.439964                                # eV·nm
IMAGE_PREFACTOR = COULOMB_CONST / 4                     # eV·nm
J_PREFACTOR_CONST = (4 * np.pi * M_E * C * KB) / (H**3) # 1 / (nm^2 · fs · K)