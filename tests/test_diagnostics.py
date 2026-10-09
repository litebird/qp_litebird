"""Tests for the pure helpers in :mod:`qp_litebird.diagnostics`."""

import unittest

import numpy as np

from qp_litebird.diagnostics import (
    apply_polarization_efficiency,
    isolate_auto_spectrum,
    relative_residual,
    zero_stokes,
)


class TestRelativeResidual(unittest.TestCase):
    def test_identical_spectra_have_zero_residual(self):
        spectrum = np.array([1.0, 2.0, 3.0, 4.0])
        max_rel, median_rel = relative_residual(spectrum, spectrum)
        self.assertEqual(max_rel, 0.0)
        self.assertEqual(median_rel, 0.0)

    def test_uniform_factor_two_is_fully_off(self):
        reference = np.ones(3)
        test = np.full(3, 2.0)
        max_rel, median_rel = relative_residual(reference, test)
        self.assertAlmostEqual(max_rel, 1.0)
        self.assertAlmostEqual(median_rel, 1.0)

    def test_uses_absolute_reference_as_denominator(self):
        reference = np.array([10.0, 10.0])
        test = np.array([10.0, 20.0])
        max_rel, median_rel = relative_residual(reference, test)
        self.assertAlmostEqual(max_rel, 1.0)
        self.assertAlmostEqual(median_rel, 0.5)

    def test_ell_min_restricts_the_comparison(self):
        reference = np.array([1.0, 1.0, 100.0])
        test = np.array([1.0, 1.0, 0.0])
        max_rel, median_rel = relative_residual(reference, test, ell_min=2)
        self.assertAlmostEqual(max_rel, 1.0)
        self.assertAlmostEqual(median_rel, 1.0)

    def test_zero_reference_does_not_divide_by_zero(self):
        reference = np.zeros(3)
        test = np.zeros(3)
        max_rel, _ = relative_residual(reference, test)
        self.assertEqual(max_rel, 0.0)

    def test_ell_min_beyond_spectrum_raises(self):
        with self.assertRaises(ValueError):
            relative_residual(np.ones(3), np.ones(3), ell_min=3)


class TestIsolateAutoSpectrum(unittest.TestCase):
    def test_keeps_only_requested_row(self):
        cl = np.arange(6 * 4, dtype=float).reshape(6, 4)
        isolated = isolate_auto_spectrum(cl, 2)
        np.testing.assert_array_equal(isolated[2], cl[2])
        np.testing.assert_array_equal(isolated[0], np.zeros(4))
        np.testing.assert_array_equal(isolated[1], np.zeros(4))
        np.testing.assert_array_equal(isolated[3:], np.zeros((3, 4)))

    def test_does_not_mutate_input(self):
        cl = np.arange(6 * 4, dtype=float).reshape(6, 4)
        before = cl.copy()
        isolate_auto_spectrum(cl, 1)
        np.testing.assert_array_equal(cl, before)


class TestZeroStokes(unittest.TestCase):
    def test_keeps_only_requested_stokes_component(self):
        alms = np.arange(3 * 5, dtype=float).reshape(3, 5).astype(complex)
        zeroed = zero_stokes(alms, 1)
        np.testing.assert_array_equal(zeroed[1], alms[1])
        np.testing.assert_array_equal(zeroed[0], np.zeros(5))
        np.testing.assert_array_equal(zeroed[2], np.zeros(5))

    def test_does_not_mutate_input(self):
        alms = np.ones((3, 5), dtype=complex)
        before = alms.copy()
        zero_stokes(alms, 0)
        np.testing.assert_array_equal(alms, before)


class TestApplyPolarizationEfficiency(unittest.TestCase):
    def test_scales_only_polarized_components(self):
        values = np.ones((3, 4), dtype=complex)
        scaled = apply_polarization_efficiency(values, 0.5)
        np.testing.assert_array_equal(scaled[0], np.ones(4))
        np.testing.assert_array_equal(scaled[1], np.full(4, 0.5))
        np.testing.assert_array_equal(scaled[2], np.full(4, 0.5))

    def test_unity_efficiency_is_identity(self):
        values = np.arange(12, dtype=float).reshape(3, 4).astype(complex)
        scaled = apply_polarization_efficiency(values, 1.0)
        np.testing.assert_array_equal(scaled, values)

    def test_does_not_mutate_input(self):
        values = np.ones((3, 4), dtype=complex)
        before = values.copy()
        apply_polarization_efficiency(values, 0.3)
        np.testing.assert_array_equal(values, before)


if __name__ == "__main__":
    unittest.main()
