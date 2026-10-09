"""Pure diagnostic helpers shared by the validation scripts.

These helpers are deliberately free of heavy dependencies (no ``smarties`` /
``qp_planck`` / ``litebird_sim``) so they can be unit-tested directly. They
implement the small pieces of bookkeeping used when comparing a qp_litebird
beam-mixing prediction against a SMARTIES map-level convolution:

* :func:`relative_residual` -- the max/median relative residual used as the
  headline agreement metric;
* :func:`isolate_auto_spectrum` -- restrict an input C_l to a single
  auto-spectrum so a comparison can be decomposed per CMB component;
* :func:`zero_stokes` -- restrict a ``(3, n_alm)`` Stokes array to a single
  component (T/E/B);
* :func:`apply_polarization_efficiency` -- bake a cross-polar efficiency into
  the polarized components of a beam ``(3, n_alm)`` array.
"""

from __future__ import annotations

import numpy as np

#: Relative residual floor, to avoid dividing by zero for (near-)null reference
#: spectra such as the BB spectrum of an r=0 realisation.
_RESIDUAL_FLOOR = 1e-30


def relative_residual(
    reference: np.ndarray,
    test: np.ndarray,
    ell_min: int = 0,
) -> tuple[float, float]:
    """Return the ``(max, median)`` relative residual of ``test`` vs ``reference``.

    The residual at multipole ``ell`` is
    ``|test[ell] - reference[ell]| / max(|reference[ell]|, 1e-30)``.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference spectrum (SMARTIES is the reference in this project).
    test : numpy.ndarray
        Spectrum under test (e.g. qp_litebird).
    ell_min : int
        First multipole to include in the comparison.

    Returns
    -------
    tuple[float, float]
        ``(max, median)`` of the relative residual over ``ell >= ell_min``.
    """
    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if ell_min >= reference.size:
        raise ValueError(
            f"ell_min ({ell_min}) must be smaller than the spectrum length "
            f"({reference.size})"
        )
    reference = reference[ell_min:]
    test = test[ell_min:]
    denominator = np.maximum(np.abs(reference), _RESIDUAL_FLOOR)
    residual = np.abs(test - reference) / denominator
    return float(np.max(residual)), float(np.median(residual))


def isolate_auto_spectrum(cl: np.ndarray, index: int) -> np.ndarray:
    """Return a copy of ``cl`` with every row except ``index`` set to zero.

    ``cl`` follows the healpy ``alm2cl`` ordering, so ``index=0`` selects TT,
    ``1`` EE, ``2`` BB and ``3`` TE.
    """
    isolated = np.zeros_like(cl)
    isolated[index] = cl[index]
    return isolated


def zero_stokes(alm_values: np.ndarray, keep_stokes: int) -> np.ndarray:
    """Return a copy of ``alm_values`` with all Stokes components but one zeroed."""
    zeroed = np.array(alm_values, copy=True)
    for stokes in range(zeroed.shape[0]):
        if stokes != keep_stokes:
            zeroed[stokes] = 0.0
    return zeroed


def apply_polarization_efficiency(beam_values: np.ndarray, rho: float) -> np.ndarray:
    """Return a copy of a ``(3, n_alm)`` beam with the polarized parts scaled by ``rho``.

    Component 0 (temperature) is left untouched; components 1 and 2 (the spin
    +/-2 combinations) are multiplied by ``rho``.
    """
    scaled = np.array(beam_values, copy=True)
    if scaled.shape[0] > 1:
        scaled[1:] *= rho
    return scaled
