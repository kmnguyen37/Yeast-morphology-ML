"""
Direct Python port of fit_ellipse.m (O. Gal, MATLAB File Exchange #3215),
used by the original thesis pipeline to fit an ellipse to each cell's
boundary via least-squares on the conic equation
    a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Ellipse:
    a: float          # semi-major/minor radius (non-tilt frame)
    b: float
    phi: float         # tilt, radians
    x0: float          # center, non-tilt frame
    y0: float
    x0_in: float        # center, original (tilted) frame
    y0_in: float
    long_axis: float
    short_axis: float
    status: str = ""


def fit_ellipse(x: np.ndarray, y: np.ndarray, orientation_tolerance: float = 1e-3) -> Ellipse | None:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) < 5:
        return None

    mean_x, mean_y = x.mean(), y.mean()
    x = x - mean_x
    y = y - mean_y

    X = np.column_stack([x**2, x * y, y**2, x, y])
    try:
        coeffs, *_ = np.linalg.lstsq(X, np.ones(len(x)), rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c, d, e = coeffs

    if min(abs(b / a) if a else np.inf, abs(b / c) if c else np.inf) > orientation_tolerance:
        orientation_rad = 0.5 * math.atan(b / (c - a)) if (c - a) != 0 else 0.0
        cos_phi, sin_phi = math.cos(orientation_rad), math.sin(orientation_rad)
        a, b, c, d, e = (
            a * cos_phi**2 - b * cos_phi * sin_phi + c * sin_phi**2,
            0.0,
            a * sin_phi**2 + b * cos_phi * sin_phi + c * cos_phi**2,
            d * cos_phi - e * sin_phi,
            d * sin_phi + e * cos_phi,
        )
        mean_x, mean_y = (
            cos_phi * mean_x - sin_phi * mean_y,
            sin_phi * mean_x + cos_phi * mean_y,
        )
    else:
        orientation_rad = 0.0
        cos_phi, sin_phi = 1.0, 0.0

    test = a * c
    if test <= 0:
        status = "Parabola found" if test == 0 else "Hyperbola found"
        return Ellipse(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, status)

    if a < 0:
        a, c, d, e = -a, -c, -d, -e

    x0 = mean_x - d / (2 * a)
    y0 = mean_y - e / (2 * c)
    F = 1 + (d**2) / (4 * a) + (e**2) / (4 * c)
    ra, rb = math.sqrt(abs(F / a)), math.sqrt(abs(F / c))
    long_axis, short_axis = 2 * max(ra, rb), 2 * min(ra, rb)

    R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    x0_in, y0_in = R @ np.array([x0, y0])

    return Ellipse(ra, rb, orientation_rad, x0, y0, x0_in, y0_in, long_axis, short_axis, "")
