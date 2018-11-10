"""
Module for everything related to accumulation rate.

With other words, the calculation of snowfall and melting rate.
"""

import numpy as np


def simple_model(
    start: float,
    stop: float,
    permanent_snow_line: float,
    slope: float,
    num: int,
    permanent_snow_rate: float,
):
    """
    Create accumulation rate array.

    :param start: x start coordinate.
    :param stop: x stop coordinate.
    :param permanent_snow_line: For x < permanent_snow_line, accumulation will
      be set equal to permanent_snow_rate.
    :param slope: The accumulation slope for x > permanent_snow_line.
    :param num: Number of points to generate.
    :permanent_snow_rate: Accumulation rate for x < permanent_snow_line.
    """
    xs = np.linspace(start, stop, num, endpoint=True)
    accumulation = np.repeat(permanent_snow_rate, num).astype(float)

    accumulation[xs >= permanent_snow_line] = permanent_snow_rate + slope * (
        xs[xs >= permanent_snow_line] - permanent_snow_line
    )
    return accumulation
