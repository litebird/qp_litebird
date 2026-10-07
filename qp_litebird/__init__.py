"""
qp_litebird
===========

QuickPol-based beam-mixing matrix computation for LiteBIRD.

This package adapts the QuickPol algorithm (E. Hivon, S. Mottet and
N. Ponthieu, "QuickPol", A&A 598, A25 (2017)) to LiteBIRD inputs: it reads
LiteBIRD h-maps (HDF5) and ``SphericalHarmonics`` beams and returns the
effective 3x3 beam-mixing matrices (W_ell) plus the corresponding mixed power
spectra, for comparison against SMARTIES map-level convolution.

Public API:
- ``hmap2mat()``      -> build the beam-mixing matrix from h-maps + beams
- ``get_spectra()``   -> mixed power spectra from a beam-mixing matrix
- ``BeamMatrixResult`` -> container returned by ``hmap2mat``
"""

from .hmap2mat import BeamMatrixResult, hmap2mat
from .spectra import get_mixed_spectrum, get_spectra

__all__ = [
    "BeamMatrixResult",
    "get_mixed_spectrum",
    "get_spectra",
    "hmap2mat",
]
