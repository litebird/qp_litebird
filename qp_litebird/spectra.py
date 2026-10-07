"""
Power-spectrum utilities for qp_litebird.

Given the beam-mixing matrix returned by :func:`qp_litebird.hmap2mat`, compute
the leakage-mixed power spectra that would be observed in a map, following the
same convention as the Planck demo.
"""

from __future__ import annotations

import healpy as hp
import numpy as np

_SPECTRUM_COMPONENTS = ["TT", "EE", "BB", "TE", "EB", "TB"]

# healpy's 6-spectrum alm2cl ordering.
_COMPONENT_TO_CL_INDEX = {
    "TT": 0,
    "EE": 1,
    "BB": 2,
    "TE": 3,
    "TB": 5,
    "EB": 4,
    "ET": 3,
    "BT": 5,
    "BE": 4,
}


def get_mixed_spectrum(beam_mat, cl_true, lmax, component):
    """Return the leakage-mixed spectrum of a single component.

    Parameters
    ----------
    beam_mat : dict[str, np.ndarray]
        Beam-mixing matrix keyed by ``TT``/``TE``/``EE``/``BB``/``TB``/``EB``,
        each of shape ``(lmax + 1, 3, 3)``.
    cl_true : np.ndarray
        Input C_ell, of shape ``(4, lmax + 1)`` or ``(6, lmax + 1)``, ordered
        like ``healpy.alm2cl`` (``[TT, EE, BB, TE]`` or
        ``[TT, EE, BB, TE, EB, TB]``).
    lmax : int
        Maximum multipole.
    component : str
        Output component, e.g. ``"TT"`` or ``"EB"``.

    Returns
    -------
    np.ndarray
        Mixed spectrum of shape ``(lmax + 1,)``.
    """
    component = component.upper()

    if cl_true.shape[0] != 6:
        if cl_true.shape[0] != 4:
            raise ValueError(
                "cl_true must have shape (4, lmax+1) or (6, lmax+1), "
                f"got {cl_true.shape}"
            )
        cl_true = np.pad(cl_true, ((0, 2), (0, 0)), mode="constant")

    cl_map = {
        comp: cl_true[_COMPONENT_TO_CL_INDEX[comp]][: lmax + 1]
        for comp in _SPECTRUM_COMPONENTS
    }

    component_to_matrix_index = {
        "TT": (0, 0),
        "TE": (0, 1),
        "TB": (0, 2),
        "ET": (1, 0),
        "EE": (1, 1),
        "EB": (1, 2),
        "BT": (2, 0),
        "BE": (2, 1),
        "BB": (2, 2),
    }
    ci, cj = component_to_matrix_index[component]

    output = np.zeros(lmax + 1)
    for source in _SPECTRUM_COMPONENTS:
        output += beam_mat[source][:, ci, cj] * cl_map[source]
    return output


def get_spectra(beam_mat, cl_true, lmax, *, pixwin=False, nside=None):
    """Return the leakage-mixed power spectra of all six components.

    Parameters
    ----------
    beam_mat : dict[str, np.ndarray]
        Beam-mixing matrix (see :func:`get_mixed_spectrum`).
    cl_true : np.ndarray
        Input C_ell in healpy's 6-spectrum order ``[TT, EE, BB, TE, EB, TB]``.
    lmax : int
        Maximum multipole.
    pixwin : bool
        If True, multiply the output by the squared HEALPix temperature pixel
        window.
    nside : int or None
        Required when ``pixwin`` is True.

    Returns
    -------
    np.ndarray
        Array of shape ``(6, lmax + 1)`` ordered ``[TT, EE, BB, TE, EB, TB]``,
        matching healpy's ``alm2cl`` / ``anafast`` convention.
    """
    window = (
        hp.pixwin(nside, pol=True)[0][: lmax + 1]
        if pixwin
        else np.ones(lmax + 1)
    )
    if pixwin and nside is None:
        raise ValueError("nside must be provided when pixwin=True")

    spectra = [
        get_mixed_spectrum(beam_mat, cl_true, lmax, component) * window**2
        for component in _SPECTRUM_COMPONENTS
    ]
    return np.array(spectra)
