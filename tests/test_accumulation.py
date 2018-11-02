"""Tests for the accumulation module."""

import numpy as np

from accumulation import simple_model


def test_permanent_snow_line():
    q = simple_model(
        start=0,
        stop=10,
        permanent_snow_line=5,
        slope=-1,
        num=11,
        permanent_snow_rate=2,
    )
    expected_q = np.array([2, 2, 2, 2, 2, 2, 1, 0, -1, -2, -3]).astype(float)
    np.testing.assert_array_equal(q, expected_q)
