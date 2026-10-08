# qp_litebird

QuickPol implementation for LiteBIRD. This code is adapted from the `qp_planck` QuickPol implementation
(https://github.com/paganol/qp_planck), itself adapted from the Planck NPIPE
pipeline. The QuickPol algorithm is described in Hivon et al. 2017 (https://arxiv.org/abs/1608.08833)

`qp_litebird` computes the effective beam-mixing matrix — the 3×3 W_ℓ windows for
TT/TE/EE/BB/TB/EB — from LiteBIRD inputs: HDF5 h-maps (`h_maps_det_*.h5`) and
`litebird_sim.SphericalHarmonics` beams, with detector metadata from
`litebird_sim.DetectorInfo`. It also provides a utility to form the mixed
power spectra from a beam matrix and true power spectra.


## Install

```bash
uv sync                      # package + runtime deps
uv sync --group notebooks    # + smarties for the notebook example
```

## Usage

```python
import qp_litebird as qp

detector_set = qp.DetectorSet(detectors, beams, h_maps_dir)  # + optional weights/rho
result = qp.hmap2mat(set_1=detector_set, set_2=None, smax=6, nside=256, lmax=512, mmax=8)

beam_mat = result.beam_mat                          # dict of (lmax+1, 3, 3) arrays
spectra = qp.get_spectra(beam_mat, cl_true, lmax)   # (6, lmax+1): TT,EE,BB,TE,EB,TB
```
A cross beam matrix between two detector can be computed using `qp.hmap2mat(set_a, set_b, ...)`.

A working example is provided in the notebook `notebooks/example.ipynb`.
