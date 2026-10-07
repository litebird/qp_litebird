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

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import healpy as hp
import numpy as np
import scipy.interpolate as inter
from litebird_sim.mapmaking.h_maps import load_h_maps_from_file

from .utilities import (
    get_beam_angles,
    get_beam_rho,
    get_beam_weights,
    spherical_harmonics_to_blm_array,
    validate_pixel_undersampling,
)

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
    weights : array or None
        Per-detector weights (default: equal weights).
    rho_beam : {"detector", "Ideal"} or array
        Cross-polar efficiency for the beam: ``"detector"`` reads
        ``detector.pol_efficiency``, ``"Ideal"`` uses 1, or pass an array.
    rho_hit : {"detector", "Ideal"} or array
        Cross-polar efficiency for the hit matrix (same options).
    """

    detectors: Sequence
    beams: Mapping
    h_maps_dir: str
    weights: np.ndarray | None = None
    rho_beam: str | np.ndarray = "Ideal"
    rho_hit: str | np.ndarray = "Ideal"

# -----------------------------------------------------------------------------
# Small linear-algebra helpers
# -----------------------------------------------------------------------------


def adjoint(matrix):
    """Hermitian adjoint (conjugate transpose)."""
    return np.transpose(np.conjugate(matrix))


def matrix_product(*matrices):
    """Chain matrix multiplications."""
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.dot(result, matrix)
    return result


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
    """Construct the beam matrix (Eq. E.4 in QuickPol) from a beam dict.

    LiteBIRD beams are always polarized, so ``ndb == 3`` and the cross-polar
    efficiency is already baked into the spin components (see
    :func:`spherical_harmonics_to_blm_array`).
    """
    mmax = beam_dict["mmax"]
    ndb = beam_dict["ndb"]
    sign = (-1.0) ** np.arange(mmax + 1)
    n = mmax
    beam = np.zeros((2 * n + 1, ndb), dtype=np.complex128)
    # m * (phi + phi0), identical for temperature and polarization
    phase_argument = np.arange(n + 1, dtype=np.float64)
    phase_argument *= beam_dict["angle"] + beam_dict["angle_shift"]
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
    rho_beam,
    weights,
    angles,
    angle_shift_deg,
):
    """Build the list of per-detector beam dictionaries consumed by the core."""
    beam_dicts = []
    for index, detector in enumerate(detectors):
        if detector.name not in beams:
            raise KeyError(
                f"No beam provided for detector {detector.name!r}"
            )
        beam_array = spherical_harmonics_to_blm_array(
            beams[detector.name], lmax, mmax, rho=rho_beam[index]
        )
        beam_dicts.append(
            {
                "det": detector.name,
                "angle": angles[index],
                "weight": weights[index],
                "lmax": lmax,
                "mmax": mmax,
                "blm": beam_array,
                "ndb": 3,
                "angle_shift": np.radians(angle_shift_deg),
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

    The LiteBIRD h-map ``h_{s,0}`` is already divided by the number of hits
    ``N_p``. QuickPol expects the *raw* moment ``sum exp(i s psi)``, so we
    multiply back by ``N_p`` here (this is the one adaptation that allows the
    hit-matrix core to stay unchanged).
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


def make_hit_vectors(
    detectors,
    weights,
    rho_hit,
    nside,
    smax,
    h_maps_dir,
    pixels,
    thr,
):
    """Build the inverse-hit-weighted spin vectors of every detector."""
    n_detectors = len(weights)
    npq, _nhigh, _skip, _sample = count_pixels(pixels)

    hit = np.zeros(npq, dtype=np.float64)
    hit_spin = np.zeros((npq, 3), dtype=np.complex128)
    hit_vectors = np.zeros((n_detectors, smax + 1, npq), dtype=np.complex128)

    rho1_weight = rho_hit * weights
    rho2_weight = rho_hit * rho1_weight

    for index, detector in enumerate(detectors):
        hit_vectors[index] = load_spin_moments(
            h_maps_dir, detector.name, nside, smax, pixels
        )
        hit += weights[index] * hit_vectors[index, 0].real
        hit_spin[:, 0] += rho2_weight[index] * hit_vectors[index, 0]
        hit_spin[:, 1] += rho1_weight[index] * hit_vectors[index, 2]
        hit_spin[:, 2] += rho2_weight[index] * hit_vectors[index, 4]

    hit = np.maximum(hit, 1.0)  # put 1 in empty pixels
    for spin in [0, 1, 2]:
        hit_spin[:, spin] /= hit
    inverse_hit, nbad = invert_hit_matrix(hit_spin, thr=thr)
    del hit_spin

    for index in range(n_detectors):
        for spin in range(smax + 1):
            hit_vectors[index, spin] /= hit
    del hit

    filtered_hit_vectors = np.zeros(
        (n_detectors, smax + 1, 3, npq), dtype=np.complex128
    )
    step = 1024 * 4
    for index in range(n_detectors):
        for spin in range(smax + 1):
            for first in range(0, npq, step):
                last = min(first + step, npq)
                hs0 = hit_vectors[index, spin, first:last]
                hsp2 = (
                    hit_vectors[index, spin + 2, first:last]
                    if spin <= smax - 2
                    else 0
                )
                hsp4 = (
                    hit_vectors[index, spin + 4, first:last]
                    if spin <= smax - 4
                    else 0
                )
                hsm2 = (
                    hit_vectors[index, spin - 2, first:last]
                    if spin >= 2
                    else np.conjugate(hit_vectors[index, abs(spin - 2), first:last])
                )
                hsm4 = (
                    hit_vectors[index, spin - 4, first:last]
                    if spin >= 4
                    else np.conjugate(hit_vectors[index, abs(spin - 4), first:last])
                )

                filtered_hit_vectors[index, spin, 0, first:last] = (
                    inverse_hit[first:last, 0, 0] * hs0
                    + rho_hit[index]
                    * (
                        inverse_hit[first:last, 0, 1] * hsp2
                        + inverse_hit[first:last, 0, 2] * hsm2
                    )
                )
                filtered_hit_vectors[index, spin, 1, first:last] = (
                    inverse_hit[first:last, 1, 0] * hsm2
                    + rho_hit[index]
                    * (
                        inverse_hit[first:last, 1, 1] * hs0
                        + inverse_hit[first:last, 1, 2] * hsm4
                    )
                )
                filtered_hit_vectors[index, spin, 2, first:last] = (
                    inverse_hit[first:last, 2, 0] * hsp2
                    + rho_hit[index]
                    * (
                        inverse_hit[first:last, 2, 1] * hsp4
                        + inverse_hit[first:last, 2, 2] * hs0
                    )
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

    ``set_2`` may be the same object as ``set_1`` (auto case); in that case the
    per-pixel hit vectors are computed once and reused, exactly as QuickPol does
    when ``detset1 == detset2``.
    """
    pixel_undersampling = validate_pixel_undersampling(pixel_undersampling)

    weights_1 = get_beam_weights(set_1.weights, len(set_1.detectors))
    weights_2 = get_beam_weights(set_2.weights, len(set_2.detectors))
    rho_hit_1 = get_beam_rho(set_1.detectors, set_1.rho_hit)
    rho_hit_2 = get_beam_rho(set_2.detectors, set_2.rho_hit)

    n_detectors_1 = len(weights_1)
    n_detectors_2 = len(weights_2)
    npix = 12 * nside**2

    if conserve_memory:
        nq = 12 if max(n_detectors_1, n_detectors_2) <= 4 else 12 * 2
    else:
        nq = 1
    npq = npix // nq

    skip = 1 if pixel_undersampling is None else pixel_undersampling
    same_set = set_2 is set_1

    hit_matrix = np.zeros(
        (n_detectors_1, n_detectors_2, smax + 1, 3, 3), dtype=np.complex128
    )
    nbad_1 = 0
    nbad_2 = 0
    npt = 0

    for iq in range(nq):
        npt += npq
        pixels = [iq * npq, (iq + 1) * npq - 1, skip]
        hit_vectors_1, nbad_chunk_1 = make_hit_vectors(
            set_1.detectors,
            weights_1,
            rho_hit_1,
            nside,
            smax,
            set_1.h_maps_dir,
            pixels,
            thr,
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
                thr,
            )
            nbad_2 += nbad_chunk_2

        step = 1024 * 4
        for first in range(0, npq, step):
            last = min(first + step, npq)
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

    hit_matrix /= npt / skip  # divide by the number of (sampled) pixels
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

    # To be consistent with CMBFAST and Healpix on TE and TB
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
        cpp[index] = matrix_product(rot, mcl, adjoint_rot)

    cout = np.zeros((n_ctypes, lmax + 1, 3, 3), dtype=np.complex128)
    sampled_ell = np.arange(0, lmax + 1, lstep)
    sampled_ell[-1] = lmax  # cannot extrapolate

    n_detectors_1 = len(beam_dicts_1)
    n_detectors_2 = len(beam_dicts_2)
    n_ell = len(sampled_ell)

    beams_1 = np.zeros(
        (2 * smax + 1, n_detectors_1, n_ell, 3, 3), dtype=np.complex128
    )
    beams_2 = np.zeros(
        (2 * smax + 1, n_detectors_2, n_ell, 3, 3), dtype=np.complex128
    )
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
                    hit_mat = matrix_product(swap12, hit_mat.conjugate(), swap12)

                for ell in sampled_ell:
                    beam_1 = beams_1[spin + smax, i1, ell // lstep]
                    beam_2 = beams_2[spin + smax, i2, ell // lstep]
                    adjoint_beam_1 = adjoint(beam_1)

                    m1 = matrix_product(idiag, adjoint_beam_1, diag)
                    m2 = matrix_product(diag, beam_2, idiag)
                    for index in range(n_ctypes):
                        m = matrix_product(m1, cpp[index], m2)
                        mh = m * hit_mat
                        mf = matrix_product(inverse_rot, mh, adjoint_inverse_rot)
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
    nside=None,
    lmax=None,
    mmax=None,
    pixel_undersampling=None,
    conserve_memory=True,
    angle_shift=0.0,
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
    nside : int or None
        HEALPix resolution; if ``None`` it is inferred from ``set_1``'s h-maps
        and checked against ``set_2``.
    lmax : int or None
        Maximum multipole; defaults to ``2 * nside``.
    mmax : int or None
        Maximum azimuthal index of the beams; defaults to ``smax + 2``.
    pixel_undersampling : int or None
        Pixel skip factor (must be a power of 4), or ``None`` for no
        undersampling.
    conserve_memory : bool
        Chunk the pixel loop to reduce peak memory.
    angle_shift : float
        Additional polarization rotation in degrees.
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

    if nside is None:
        nside = infer_nside(set_1.h_maps_dir, names_1[0])
    if lmax is None:
        lmax = 2 * nside
    if mmax is None:
        mmax = smax + 2

    if set_2 is not set_1:
        nside_2 = infer_nside(set_2.h_maps_dir, names_2[0])
        if nside_2 != nside:
            raise ValueError(
                f"set_2 h-maps nside ({nside_2}) does not match set_1 nside "
                f"({nside}); a cross hit matrix requires a common resolution."
            )

    thr = 3.0e-3
    ctypes = ["TT", "TE", "EE", "BB", "TB", "EB"]

    beam_dicts_1 = fill_beam_dict(
        set_1.detectors,
        set_1.beams,
        lmax,
        mmax,
        get_beam_rho(set_1.detectors, set_1.rho_beam),
        get_beam_weights(set_1.weights, len(set_1.detectors)),
        get_beam_angles(set_1.detectors),
        angle_shift,
    )
    if set_2 is set_1:
        beam_dicts_2 = beam_dicts_1
    else:
        beam_dicts_2 = fill_beam_dict(
            set_2.detectors,
            set_2.beams,
            lmax,
            mmax,
            get_beam_rho(set_2.detectors, set_2.rho_beam),
            get_beam_weights(set_2.weights, len(set_2.detectors)),
            get_beam_angles(set_2.detectors),
            angle_shift,
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
    beam_mat = {
        ctype: beam_windows[index].real for index, ctype in enumerate(ctypes)
    }

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
