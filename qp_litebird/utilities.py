"""
Utility helpers for qp_litebird.

This module is intentionally small: it only contains the pieces that convert
LiteBIRD ``SphericalHarmonics`` beams into the ``(lmax + 1, mmax + 1, 3)``
array layout consumed by the QuickPol core, together with a couple of tiny
validation helpers.

The beam conversion reproduces, algebraically, the FITS round-trip that the
Planck demo relied on:

    convert_alm_plusminus_to_spin(E, B, spin=2)   # smarties.utils.harmonics
        -> write_blm_fits(..., balm_convertion=False)   # demo's tools.py
        -> get_blm_det(..., isbalm=True)                # qp_planck

Collapsing those three steps gives, for a polarized beam ``(T, E, B)``::

    ret[l, m, 0] =  T[l, m]                  * norm
    ret[l, m, 1] = -(E[l, m] + 1j * B[l, m]) * norm * rho
    ret[l, m, 2] = -(E[l, m] - 1j * B[l, m]) * norm * rho

with ``norm = 1 / (T[0, 0] * sqrt(2 l + 1))``. The cross-polar efficiency
``rho`` is baked into the spin components because QuickPol's ``bmat`` only
applies ``rho`` in its scalar (``ndb == 1``) branch; for polarized beams
(``ndb == 3``) the efficiency must already be inside the beam coefficients.
"""

from __future__ import annotations

import healpy as hp
import numpy as np


def spherical_harmonics_to_blm_array(beam, lmax, mmax):
    """Convert a ``SphericalHarmonics`` beam to the QuickPol blm array.

    Parameters
    ----------
    beam : litebird_sim.SphericalHarmonics
        Beam with ``.values`` of shape ``(3, n_alm)`` (T, E, B) and
        ``.lmax`` / ``.mmax`` attributes.
    lmax : int
        Maximum multipole of the output array.
    mmax : int
        Maximum azimuthal index of the output array.
    rho : float, optional
        Cross-polar efficiency multiplying the polarized components.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(lmax + 1, mmax + 1, 3)`` in the layout expected by
        ``bmat`` (component 0 = temperature, 1 and 2 = spin +-2).
    """
    if beam.mmax < mmax:
        raise ValueError(
            f"Beam mmax ({beam.mmax}) is smaller than the requested mmax ({mmax}); "
            "resize the beam or lower mmax."
        )
    if beam.lmax < lmax:
        raise ValueError(
            f"Beam lmax ({beam.lmax}) is smaller than the requested lmax ({lmax})."
        )

    values = beam.values  # shape (3, n_alm)
    beam_lmax = beam.lmax

    ret = np.zeros((lmax + 1, mmax + 1, 3), dtype=np.complex128)
    for l in range(lmax + 1):
        for m in range(min(l, mmax) + 1):
            idx = hp.Alm.getidx(beam_lmax, l, m)
            temperature = values[0, idx]
            e_mode = values[1, idx]
            b_mode = values[2, idx]
            ret[l, m, 0] = temperature
            ret[l, m, 1] = -(e_mode + 1j * b_mode)
            ret[l, m, 2] = -(e_mode - 1j * b_mode)

    # "balm" renormalization: divide by b_00 and by sqrt(2l + 1).
    normalization = ret[0, 0, 0]
    for l in range(lmax + 1):
        ret[l, :, :] *= 1.0 / (normalization * np.sqrt(2.0 * l + 1.0))

    return ret


def validate_pixel_undersampling(pixel_undersampling):
    """Validate the pixel-undersampling factor (``None`` or a power of 4)."""
    if pixel_undersampling is None:
        return None

    if isinstance(pixel_undersampling, np.generic):
        pixel_undersampling = int(pixel_undersampling)

    if not isinstance(pixel_undersampling, int):
        raise TypeError(
            "pixel_undersampling must be None or an integer power of 4 "
            "(e.g. 1, 4, 16, 64, ...)."
        )
    if pixel_undersampling < 1:
        raise ValueError("pixel_undersampling must be >= 1.")

    factor = pixel_undersampling
    while factor % 4 == 0:
        factor //= 4
    if factor != 1:
        raise ValueError(
            "pixel_undersampling must be a power of 4 (1, 4, 16, 64, ...)."
        )

    return pixel_undersampling
