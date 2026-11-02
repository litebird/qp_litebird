"""
QuickPol driver for LiteBIRD beam-mixing matrices.

This module adapts the QuickPol beam-matrix algorithm (originally from the
Planck NPIPE pipeline, see E. Hivon, S. Mottet and N. Ponthieu, "QuickPol",
A&A 598, A25 (2017)) to LiteBIRD inputs:

  * h-maps are read with ``litebird_sim.mapmaking.h_maps.load_h_maps_from_file``
    (HDF5, one file per detector). LiteBIRD h-maps are already normalized by the
    number of hits (``h_{0,0} = N_p`` and ``h_{s,0} = (1/N_p) sum exp(i s psi)``),
    so the reader un-normalizes them (``h_{s,0} * N_p``) to recover the raw spin
    moments that the QuickPol hit-matrix core expects. This keeps the core
    computation unchanged.
  * beams are ``litebird_sim.SphericalHarmonics`` objects converted to the
    ``(lmax + 1, mmax + 1, 3)`` layout consumed by ``build_beam_matrix``.
  * detector metadata (polarization angle, cross-polar efficiency) comes from
    ``litebird_sim.DetectorInfo``.

Inputs are bundled into a :class:`DetectorSet` (detectors, beams, h-map
directory, weights, rho). The top-level entry point is :func:`hmap2mat`, which
accepts one or two sets: with a single set it computes the auto beam matrix, and
with a second set it reproduces QuickPol's ``detset1 x detset2`` cross treatment.
It returns a :class:`BeamMatrixResult`.
"""

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import healpy as hp
import litebird_sim as lbs
import numpy as np
import scipy.interpolate as inter
from litebird_sim.mapmaking.h_maps import load_h_maps_from_file

from .utilities import (
    spherical_harmonics_to_blm_array,
    validate_pixel_undersampling,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Detector sets
# -----------------------------------------------------------------------------


@dataclass
class DetectorSet:
    """A LiteBIRD detector set and its per-set inputs.

    Parameters
    ----------
    detectors : sequence of litebird_sim.DetectorInfo
        Detectors in the set. All are assumed to be polarization-sensitive.
    beams : mapping[str, litebird_sim.SphericalHarmonics]
        Beam harmonic coefficients keyed by detector name. Each beam ``mmax``
        must be at least ``smax + 2``.
    h_maps_dir : str
        Directory containing the ``h_maps_det_{name}.h5`` files.
    nside : int or None
        Optional nside of the h-maps. If None, it is inferred from the first detector's h-maps.
    weights : {"imo", "equal"} or array
        Per-detector weights (default: equal).
    rho_hit : {"imo", "ideal"} or array
        Polarisation efficiency assumed by the hit matrix (default: ideal).
    path_mask_T : str or Path or None
        Optional path to a temperature mask. The mask solely affects which pixels are taken in consideration to build the hit matrix, it doesn't model the cut-sky mode coupling.
    path_mask_P : str or Path or None
        Optional path to a polarization mask. The mask solely affects which pixels are taken in consideration to build the hit matrix, it doesn't model the cut-sky mode coupling. If only one mask is provided, it will be used for both temperature and polarization.
    """

    detectors: Sequence[lbs.DetectorInfo]
    beams: Mapping[str, lbs.SphericalHarmonics]
    h_maps_dir: str | Path
    nside: int | None = None
    weights: np.ndarray | str = "equal"
    rho_hit: np.ndarray | str = "ideal"
    path_mask_T: str | Path | None = None
    path_mask_P: str | Path | None = None

    mask_T: np.ndarray | None = field(init=False, default=None)
    mask_P: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self):
        if isinstance(self.weights, str):
            if (self.weights).lower() == "imo":
                self.weights = np.array(
                    [1 / (det.net_ukrts) ** 2 for det in self.detectors]
                )
            elif (self.weights).lower() == "equal":
                self.weights = np.ones(len(self.detectors))
            else:
                raise ValueError(
                    f"weights must be 'imo', 'equal', or an array; got {self.weights!r}"
                )
        if isinstance(self.rho_hit, str):
            if (self.rho_hit).lower() == "imo":
                self.rho_hit = np.array([det.pol_efficiency for det in self.detectors])
            elif (self.rho_hit).lower() == "ideal":
                self.rho_hit = np.ones(len(self.detectors))
            else:
                raise ValueError(
                    f"rho_hit must be 'imo', 'ideal', or an array; got {self.rho_hit!r}"
                )

        assert len(self.detectors) == (self.weights).size == (self.rho_hit).size, (
            "detectors, weights, and rho_hit must have the same length."
        )
        assert self.weights.ndim == self.rho_hit.ndim == 1, (
            "weights and rho_hit must be 1D arrays."
        )
        if self.path_mask_P is not None:
            self.path_mask_P = Path(self.path_mask_P)
        if self.path_mask_T is not None:
            self.path_mask_T = Path(self.path_mask_T)

        self.h_maps_dir = Path(self.h_maps_dir)

        assert self.h_maps_dir.is_dir(), (
            f"h_maps_dir {self.h_maps_dir} is not a directory."
        )
        assert self.path_mask_T is None or self.path_mask_T.is_file(), (
            f"Temperature mask {self.path_mask_T} is not a file."
        )
        assert self.path_mask_P is None or self.path_mask_P.is_file(), (
            f"Polarization mask {self.path_mask_P} is not a file."
        )

        if self.nside is None:
            self.nside = infer_nside(self.h_maps_dir, self.detectors[0].name)

    def load_masks(self):
        """Load the temperature and polarization masks, if provided."""
        mask_T = None
        mask_P = None
        match (self.path_mask_T, self.path_mask_P):
            case (None, None):
                pass
            case (Path() as p, None):
                mask_T = hp.read_map(p)
                mask_P = mask_T
            case (None, Path() as p):
                mask_P = hp.read_map(p)
                mask_T = mask_P
            case (Path() as pT, Path() as pP):
                mask_T = hp.read_map(pT)
                mask_P = hp.read_map(pP)
            case _:
                raise ValueError(
                    "path_mask_T and path_mask_P must be None or Path objects."
                )
        if mask_T is not None and mask_P is not None:
            if hp.get_nside(mask_T) != self.nside:
                logger.warning("Converting temperature mask to nside=%d", self.nside)
                mask_T = hp.ud_grade(mask_T, nside_out=self.nside, power=0)
            if hp.get_nside(mask_P) != self.nside:
                logger.warning("Converting polarization mask to nside=%d", self.nside)
                mask_P = hp.ud_grade(mask_P, nside_out=self.nside, power=0)
        self.mask_T = mask_T
        self.mask_P = mask_P


# -----------------------------------------------------------------------------
# Small linear-algebra helpers
# -----------------------------------------------------------------------------


def adjoint(matrix):
    """Hermitian adjoint (conjugate transpose)."""
    return np.transpose(np.conjugate(matrix))


def interpolate_matrix(beam_windows, lmax, sampled_ell):
    """Interpolate a matrix in ell using cubic interpolation."""
    n1 = beam_windows.shape[1]
    n2 = beam_windows.shape[2]
    is_complex = isinstance(beam_windows[0, 0, 0], np.complex128)
    output = np.zeros((lmax + 1, n1, n2), dtype=beam_windows.dtype)
    ell = np.arange(lmax + 1)
    for i in range(n1):
        for j in range(n2):
            values = beam_windows[sampled_ell, i, j]
            if is_complex:
                imag_interp = inter.interp1d(sampled_ell, values.imag, kind="cubic")
                output[ell, i, j] += imag_interp(ell) * 1j
            real_interp = inter.interp1d(sampled_ell, values.real, kind="cubic")
            output[ell, i, j] += real_interp(ell)
    return output


# -----------------------------------------------------------------------------
# Beam matrices
# -----------------------------------------------------------------------------


def build_beam_matrix(beam_dict, ell, spin, verbose=False):
    """Construct the beam matrix (Eq. E.4 in QuickPol) from a beam dict."""
    mmax = beam_dict["mmax"]
    sign = (-1.0) ** np.arange(mmax + 1)
    n = mmax
    beam = np.zeros((2 * n + 1, 3), dtype=np.complex128)
    # m * (phi + phi0), identical for temperature and polarization
    phase_argument = np.arange(n + 1, dtype=np.float64)
    phase_argument *= beam_dict["angle"]
    phase = np.cos(phase_argument) + 1j * np.sin(phase_argument)

    temperature = beam_dict["blm"][ell, 0 : n + 1, 0] * phase
    beam[n : 2 * n + 1, 0] = temperature[0 : n + 1]
    beam[n - 1 :: -1, 0] = np.conj((temperature * sign)[1 : n + 1])

    spin_1 = beam_dict["blm"][ell, 0 : n + 1, 1] * phase
    spin_2 = beam_dict["blm"][ell, 0 : n + 1, 2] * phase
    beam[n : 2 * n + 1, 1] = spin_1[0 : n + 1]
    beam[n : 2 * n + 1, 2] = spin_2[0 : n + 1]
    beam[n - 1 :: -1, 1] = np.conj((spin_2 * sign)[1 : n + 1])
    beam[n - 1 :: -1, 2] = np.conj((spin_1 * sign)[1 : n + 1])

    beam_matrix = np.array(
        [
            [beam[n + spin, 0], beam[n + spin - 2, 0], beam[n + spin + 2, 0]],
            [beam[n + spin, 1], beam[n + spin - 2, 1], beam[n + spin + 2, 1]],
            [beam[n + spin, 2], beam[n + spin - 2, 2], beam[n + spin + 2, 2]],
        ]
    )
    beam_matrix *= beam_dict["weight"]
    return beam_matrix


def fill_beam_dict(
    detectors,
    beams,
    lmax,
    mmax,
    weights,
):
    """Build the list of per-detector beam dictionaries."""
    beam_dicts = []
    for index, detector in enumerate(detectors):
        if detector.name not in beams:
            raise KeyError(f"No beam provided for detector {detector.name!r}")
        beam_array = spherical_harmonics_to_blm_array(beams[detector.name], lmax, mmax)
        beam_dicts.append(
            {
                "det": detector.name,
                "angle": detector.pol_angle_rad,
                "weight": weights[index],
                "lmax": lmax,
                "mmax": mmax,
                "blm": beam_array,
            }
        )
    return beam_dicts


# -----------------------------------------------------------------------------
# Hit matrices
# -----------------------------------------------------------------------------


def count_pixels(pixels):
    """Convert a pixel slice ``[start, stop, (optional) step]`` to counts."""
    if pixels is None or len(pixels) < 2 or len(pixels) > 3:
        raise ValueError(f"Invalid pixels: {pixels}")
    nhigh = pixels[1] - pixels[0] + 1
    skip = pixels[2] if len(pixels) == 3 else 1
    sample = skip > 1
    nlow = (nhigh + skip - 1) // skip
    return nlow, nhigh, skip, sample


def load_spin_moments(h_maps_dir, detector_name, nside, smax, pixels):
    """Load the hit count and un-normalized spin moments of one detector.

    The LiteBIRD h-map ``h_{s,0}`` is divided by the number of hits
    ``N_p``. QuickPol uses ``sum exp(i s psi)``, so we
    multiply back by ``N_p``
    """
    path = os.path.join(h_maps_dir, f"h_maps_det_{detector_name}.h5")
    result = load_h_maps_from_file(path)
    h_maps = result.h_maps[detector_name]

    hit = np.array(h_maps[(0, 0)].real, dtype=np.float64)
    nside_hit = hp.npix2nside(hit.size)
    if nside != nside_hit:
        raise ValueError(
            f"h-map nside ({nside_hit}) does not match the requested nside "
            f"({nside}); LiteBIRD h-maps are expected at the target resolution."
        )

    nlow, _nhigh, skip, sample = count_pixels(pixels)
    spin_moments = np.zeros((smax + 1, nlow), dtype=np.complex128)
    for spin in range(smax + 1):
        if spin == 0:
            buffer = hit
        else:
            h_map = h_maps[(spin, 0)]
            buffer = (h_map.real + 1j * h_map.imag) * hit
        buffer = buffer[pixels[0] : pixels[1] + 1]
        spin_moments[spin] = buffer[::skip] if sample else buffer
    return spin_moments


def invert_hit_submatrix(matrix, thr=1.0e-3):
    """Analytically invert the 3x3 hit sub-matrix (polarized case)."""
    n = matrix.shape[0]
    output = np.zeros((n, 3, 3), dtype=np.complex128)

    x = matrix[:, 0].real
    z2 = matrix[:, 1]
    z4 = matrix[:, 2]

    rho2 = np.conjugate(z2) * z2
    rho4 = np.conjugate(z4) * z4
    xm = z2 * np.conjugate(z4)
    t1 = xm - x * np.conjugate(z2)
    t1c = np.conjugate(t1)
    t2 = z2 * z2 - z4
    t2c = np.conjugate(t2)
    det = np.real(x**2 - rho4 + 2 * ((z2 * xm).real - x * rho2))

    output[:, 0, 0] = (x**2 - rho4) / det
    output[:, 0, 1] = t1 / det
    output[:, 0, 2] = t1c / det
    output[:, 1, 0] = t1c / det
    output[:, 1, 1] = (x - rho2) / det
    output[:, 1, 2] = t2 / det
    output[:, 2, 0] = t1 / det
    output[:, 2, 1] = t2c / det
    output[:, 2, 2] = output[:, 1, 1]

    bad = np.where((np.abs(det) < thr) + np.isnan(det))
    output[bad, :, :] = 0.0
    return output, bad[0].size


def invert_hit_matrix(matrix, thr=1.0e-3):
    """Blockwise inversion of the hit matrix for numerical stability."""
    n = matrix.shape[0]
    nbad = 0
    output = np.zeros((n, 3, 3), dtype=np.complex128)
    step = 1024 * 16
    for first in range(0, n, step):
        last = min(first + step, n)
        output[first:last], nbad_chunk = invert_hit_submatrix(
            matrix[first:last], thr=thr
        )
        nbad += nbad_chunk
    return output, nbad


def apply_mask_hit_matrix(inverse_hit, mask_T, mask_P, pixels):
    """Apply temperature and polarization masks to the inverse hit matrix."""
    _, _, skip, _ = count_pixels(pixels)
    if mask_T is not None:
        mask_T_chunk = mask_T[pixels[0] : pixels[1] + 1 : skip]

        if mask_P is not None:
            mask_P_chunk = mask_P[pixels[0] : pixels[1] + 1 : skip]
        else:
            mask_P_chunk = mask_T_chunk

        for k in range(3):
            inverse_hit[:, 0, k] *= mask_T_chunk
            inverse_hit[:, 1, k] *= mask_P_chunk
            inverse_hit[:, 2, k] *= mask_P_chunk

    return inverse_hit


def make_hit_vectors(
    detectors,
    weights,
    rho_hit,
    nside,
    smax,
    h_maps_dir,
    pixels,
    mask_T,
    mask_P,
    thr,
):
    """Build hit vectors and their inverse-weighted versions."""
    n_detectors = len(weights)
    n_pixels_per_chunk, _nhigh, _skip, _sample = count_pixels(pixels)
    assert np.all(weights > 0), "All weights must be positive."
    hit = np.zeros(n_pixels_per_chunk, dtype=np.float64)
    hit_spin = np.zeros((n_pixels_per_chunk, 3), dtype=np.complex128)
    hit_vectors = np.zeros(
        (n_detectors, smax + 1, n_pixels_per_chunk), dtype=np.complex128
    )

    rho1_weight = rho_hit * weights  # rho * weight
    rho2_weight = rho_hit * rho1_weight  # rho**2 * weight

    for idx_det, detector in enumerate(detectors):
        hit_vectors[idx_det] = load_spin_moments(
            h_maps_dir, detector.name, nside, smax, pixels
        )
        hit += weights[idx_det] * hit_vectors[idx_det, 0].real
        hit_spin[:, 0] += rho2_weight[idx_det] * hit_vectors[idx_det, 0]  # T
        hit_spin[:, 1] += rho1_weight[idx_det] * hit_vectors[idx_det, 2]  # Q+iU
        hit_spin[:, 2] += rho2_weight[idx_det] * hit_vectors[idx_det, 4]  # Q-iU

    valid = hit > 0.0 # select pixels with non-zero hit counts
    for comp in [0, 1, 2]:
        np.divide(hit_spin[:, comp], hit, out=hit_spin[:, comp], where=valid)
    inverse_hit, nbad = invert_hit_matrix(hit_spin, thr=thr)
    del hit_spin

    inverse_hit = apply_mask_hit_matrix(inverse_hit, mask_T, mask_P, pixels)

    for idx_det in range(n_detectors):
        for spin in range(smax + 1):
            np.divide(hit_vectors[idx_det, spin], hit, out=hit_vectors[idx_det, spin], where=valid)
    del hit

    filtered_hit_vectors = np.zeros(
        (n_detectors, smax + 1, 3, n_pixels_per_chunk), dtype=np.complex128
    )
    step = 1024 * 4
    for idx_det in range(n_detectors):
        for spin in range(smax + 1):
            for first in range(0, n_pixels_per_chunk, step):
                last = min(first + step, n_pixels_per_chunk)
                h_spin_0 = hit_vectors[idx_det, spin, first:last]
                h_spin_p2 = (
                    hit_vectors[idx_det, spin + 2, first:last]
                    if spin <= smax - 2
                    else 0
                )
                h_spin_p4 = (
                    hit_vectors[idx_det, spin + 4, first:last]
                    if spin <= smax - 4
                    else 0
                )
                h_spin_m2 = (
                    hit_vectors[idx_det, spin - 2, first:last]
                    if spin >= 2
                    else np.conjugate(hit_vectors[idx_det, abs(spin - 2), first:last])
                )
                h_spin_m4 = (
                    hit_vectors[idx_det, spin - 4, first:last]
                    if spin >= 4
                    else np.conjugate(hit_vectors[idx_det, abs(spin - 4), first:last])
                )
                #  \tilde\omega_s[0]
                filtered_hit_vectors[idx_det, spin, 0, first:last] = inverse_hit[
                    first:last, 0, 0
                ] * h_spin_0 + rho_hit[idx_det] * (
                    inverse_hit[first:last, 0, 1] * h_spin_p2
                    + inverse_hit[first:last, 0, 2] * h_spin_m2
                )
                #  \rho\tilde\omega_s[2]
                filtered_hit_vectors[idx_det, spin, 1, first:last] = inverse_hit[
                    first:last, 1, 0
                ] * h_spin_m2 + rho_hit[idx_det] * (
                    inverse_hit[first:last, 1, 1] * h_spin_0
                    + inverse_hit[first:last, 1, 2] * h_spin_m4
                )
                #  \rho\tilde\omega_s[-2]
                filtered_hit_vectors[idx_det, spin, 2, first:last] = inverse_hit[
                    first:last, 2, 0
                ] * h_spin_p2 + rho_hit[idx_det] * (
                    inverse_hit[first:last, 2, 1] * h_spin_p4
                    + inverse_hit[first:last, 2, 2] * h_spin_0
                )
    del hit_vectors
    return filtered_hit_vectors, nbad


def make_hit_matrix(
    set_1,
    set_2,
    nside,
    smax,
    pixel_undersampling=None,
    thr=None,
    conserve_memory=True,
):
    """Compute the cross hit matrix between two detector sets.

    ``set_2`` may be the same object as ``set_1``, in that case the
    per-pixel hit vectors are computed once and reused
    """
    pixel_undersampling = validate_pixel_undersampling(pixel_undersampling)

    weights_1 = set_1.weights
    weights_2 = set_2.weights
    rho_hit_1 = set_1.rho_hit
    rho_hit_2 = set_2.rho_hit

    n_detectors_1 = len(weights_1)
    n_detectors_2 = len(weights_2)
    npix = 12 * nside**2

    if conserve_memory:
        number_of_chunks = 12 if max(n_detectors_1, n_detectors_2) <= 4 else 12 * 2
    else:
        number_of_chunks = 1
    n_pixel_per_chunk = npix // number_of_chunks

    skip = 1 if pixel_undersampling is None else pixel_undersampling
    same_set = set_2 is set_1

    hit_matrix = np.zeros(
        (n_detectors_1, n_detectors_2, smax + 1, 3, 3), dtype=np.complex128
    )
    nbad_1 = 0
    nbad_2 = 0
    pixel_start_chunk = 0

    set_1.load_masks()
    set_2.load_masks()

    for idx_chunk in range(number_of_chunks):
        pixel_start_chunk += n_pixel_per_chunk
        pixels = [
            idx_chunk * n_pixel_per_chunk,
            (idx_chunk + 1) * n_pixel_per_chunk - 1,
            skip,
        ]

        hit_vectors_1, nbad_chunk_1 = make_hit_vectors(
            set_1.detectors,
            weights_1,
            rho_hit_1,
            nside,
            smax,
            set_1.h_maps_dir,
            pixels,
            mask_T=set_1.mask_T,
            mask_P=set_1.mask_P,
            thr=thr,
        )
        nbad_1 += nbad_chunk_1
        if same_set:
            hit_vectors_2 = hit_vectors_1
            nbad_2 += nbad_chunk_1
        else:
            hit_vectors_2, nbad_chunk_2 = make_hit_vectors(
                set_2.detectors,
                weights_2,
                rho_hit_2,
                nside,
                smax,
                set_2.h_maps_dir,
                pixels,
                mask_T=set_2.mask_T,
                mask_P=set_2.mask_P,
                thr=thr,
            )
            nbad_2 += nbad_chunk_2

        step = 1024 * 4
        for first in range(0, n_pixel_per_chunk, step):
            last = min(first + step, n_pixel_per_chunk)
            for i1 in range(n_detectors_1):
                for i2 in range(n_detectors_2):
                    for spin in range(smax + 1):
                        for u1 in range(3):
                            for u2 in range(3):
                                # sum_p h1 . conj(h2)
                                hit_matrix[i1, i2, spin, u1, u2] += np.vdot(
                                    hit_vectors_2[i2, spin, u2, first:last],
                                    hit_vectors_1[i1, spin, u1, first:last],
                                )

    hit_matrix /= pixel_start_chunk / skip  # divide by the number of (sampled) pixels
    return hit_matrix, nbad_1, nbad_2, skip


# -----------------------------------------------------------------------------
# Beam-window computation
# -----------------------------------------------------------------------------


def compute_beam_matrices(
    beam_dicts_1,
    beam_dicts_2,
    hit_matrix,
    lmax,
    smax,
    ctypes,
    lstep=1,
):
    """Compute the effective beam matrices for a set of input C_ell types."""
    diag = np.array([[1, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    idiag = np.array([[1, 0, 0], [0, 2.0, 0], [0, 0, 2.0]])

    # To be consistent with Healpix on TE and TB
    rot = np.array([[1, 0, 0], [0, -1, -1j], [0, -1, 1j]])
    adjoint_rot = adjoint(rot)
    inverse_rot = np.linalg.inv(rot)
    adjoint_inverse_rot = adjoint(inverse_rot)
    swap12 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    # input C(l) selector
    mcl_in = np.array([["TT", "TE", "TB"], ["TE", "EE", "EB"], ["TB", "EB", "BB"]])
    n_ctypes = len(ctypes)
    cpp = np.zeros((n_ctypes, 3, 3), dtype=np.complex128)
    for index, ctype in enumerate(ctypes):
        mcl = (mcl_in == ctype) * 1.0
        cpp[index] = rot @ mcl @ adjoint_rot

    cout = np.zeros((n_ctypes, lmax + 1, 3, 3), dtype=np.complex128)
    sampled_ell = np.arange(0, lmax + 1, lstep)
    sampled_ell[-1] = lmax  # cannot extrapolate

    n_detectors_1 = len(beam_dicts_1)
    n_detectors_2 = len(beam_dicts_2)
    n_ell = len(sampled_ell)

    beams_1 = np.zeros((2 * smax + 1, n_detectors_1, n_ell, 3, 3), dtype=np.complex128)
    beams_2 = np.zeros((2 * smax + 1, n_detectors_2, n_ell, 3, 3), dtype=np.complex128)
    for i1 in range(n_detectors_1):
        for spin in np.arange(-smax, smax + 1):
            for ell in sampled_ell:
                beams_1[spin + smax, i1, ell // lstep] = build_beam_matrix(
                    beam_dicts_1[i1], ell, spin, verbose=(ell + abs(spin) < 1)
                )
    for i2 in range(n_detectors_2):
        for spin in np.arange(-smax, smax + 1):
            for ell in sampled_ell:
                beams_2[spin + smax, i2, ell // lstep] = build_beam_matrix(
                    beam_dicts_2[i2], ell, spin, verbose=(ell + abs(spin) < 1)
                )

    for spin in np.arange(-smax, smax + 1):
        for i1 in range(n_detectors_1):
            for i2 in range(n_detectors_2):
                hit_mat = hit_matrix[i1, i2, abs(spin)]
                if spin < 0:
                    hit_mat = swap12 @ hit_mat.conjugate() @ swap12

                for ell in sampled_ell:
                    beam_1 = beams_1[spin + smax, i1, ell // lstep]
                    beam_2 = beams_2[spin + smax, i2, ell // lstep]
                    adjoint_beam_1 = adjoint(beam_1)

                    m1 = idiag @ adjoint_beam_1 @ diag
                    m2 = diag @ beam_2 @ idiag
                    for index in range(n_ctypes):
                        m = m1 @ cpp[index] @ m2
                        mh = m * hit_mat
                        mf = inverse_rot @ mh @ adjoint_inverse_rot
                        cout[index, ell] += mf

    if lstep > 1:
        for index in range(n_ctypes):
            cout[index] = interpolate_matrix(cout[index], lmax, sampled_ell)

    return cout


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------


@dataclass
class BeamMatrixResult:
    """Result of :func:`hmap2mat`.

    ``beam_mat`` maps each CMB type (``TT``, ``TE``, ``EE``, ``BB``, ``TB``,
    ``EB``) to a real array of shape ``(lmax + 1, 3, 3)``, i.e. the 3x3
    beam-mixing window at every multipole.

    For a cross computation between two detector sets, ``hit_mat`` has shape
    ``(n_det_1, n_det_2, smax + 1, 3, 3)`` and ``beam_mat`` is the effective
    beam summed over the cross pairs ``(i1, i2)``. In the auto case
    (``set_2 is set_1``) the two name lists and bad-pixel counts coincide.
    """

    beam_mat: Mapping[str, np.ndarray]
    hit_mat: np.ndarray
    nside: int
    lmax: int
    smax: int
    mmax: int
    detector_names_1: list
    detector_names_2: list
    skip: int
    nbad_1: int
    nbad_2: int


def infer_nside(h_maps_dir, detector_name):
    """Infer ``nside`` from the (0,0) h-map of one detector."""
    import h5py

    path = os.path.join(h_maps_dir, f"h_maps_det_{detector_name}.h5")
    with h5py.File(path, "r") as h5_file:
        npix = h5_file["0,0"]["Re"].shape[0]
    return hp.npix2nside(npix)


def hmap2mat(
    set_1,
    set_2=None,
    *,
    smax=6,
    lmax=None,
    mmax=None,
    pixel_undersampling=None,
    conserve_memory=True,
    lstep=1,
    savefile=None,
):
    """Compute the QuickPol beam-mixing matrix for LiteBIRD detector set(s).

    Parameters
    ----------
    set_1 : DetectorSet
        First detector set (detectors, beams, h-map directory, weights, rho).
    set_2 : DetectorSet or None
        Optional second detector set. If ``None`` the auto case is computed
        (``set_2 = set_1``). Otherwise the effective beam for the cross-spectra
        between the two sets is returned.
    smax : int
        Maximum spin in the hit/moment expansion (default 6).
    lmax : int or None
        Maximum multipole; defaults to ``2 * nside``.
    mmax : int or None
        Maximum azimuthal index of the beams; defaults to ``smax + 2``.
    pixel_undersampling : int or None
        Pixel skip factor (must be a power of 4), or ``None`` for no
        undersampling.
    conserve_memory : bool
        Chunk the pixel loop to reduce peak memory.
    lstep : int
        Multipole stride used to sample the beam matrices; cubic interpolation
        recovers the intermediate ell when ``lstep > 1``.
    savefile : str or None
        Optional path for saving ``beam_mat`` / ``hit_mat`` to an ``.npz``.

    Returns
    -------
    BeamMatrixResult
    """

    if not set_1.detectors:
        raise ValueError("set_1 has no detectors.")
    if set_2 is None:
        set_2 = set_1

    names_1 = [detector.name for detector in set_1.detectors]
    names_2 = [detector.name for detector in set_2.detectors]

    if set_2 is not set_1:
        assert set_2.nside == set_1.nside, "set_2 nside must match set_1 nside"

    nside = set_1.nside
    if lmax is None:
        lmax = 2 * nside
    if mmax is None:
        mmax = smax + 2

    thr = 3.0e-3
    ctypes = ["TT", "TE", "EE", "BB", "TB", "EB"]

    beam_dicts_1 = fill_beam_dict(
        set_1.detectors,
        set_1.beams,
        lmax,
        mmax,
        set_1.weights,
    )
    if set_2 is set_1:
        beam_dicts_2 = beam_dicts_1
    else:
        beam_dicts_2 = fill_beam_dict(
            set_2.detectors,
            set_2.beams,
            lmax,
            mmax,
            set_2.weights,
        )

    hit_matrix, nbad_1, nbad_2, skip = make_hit_matrix(
        set_1,
        set_2,
        nside,
        smax,
        pixel_undersampling=pixel_undersampling,
        thr=thr,
        conserve_memory=conserve_memory,
    )

    beam_windows = compute_beam_matrices(
        beam_dicts_1, beam_dicts_2, hit_matrix, lmax, smax, ctypes, lstep=lstep
    )
    beam_mat = {ctype: beam_windows[index].real for index, ctype in enumerate(ctypes)}

    if savefile is not None:
        directory = os.path.dirname(savefile)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        np.savez(
            savefile,
            beam_mat=beam_mat,
            hit_mat=hit_matrix,
            lmax=lmax,
            smax=smax,
            mmax=mmax,
            nside=nside,
            detector_names_1=np.array(names_1),
            detector_names_2=np.array(names_2),
            nbad_1=nbad_1,
            nbad_2=nbad_2,
        )

    return BeamMatrixResult(
        beam_mat=beam_mat,
        hit_mat=hit_matrix,
        nside=nside,
        lmax=lmax,
        smax=smax,
        mmax=mmax,
        detector_names_1=names_1,
        detector_names_2=names_2,
        skip=skip,
        nbad_1=nbad_1,
        nbad_2=nbad_2,
    )
