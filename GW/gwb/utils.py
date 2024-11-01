import numba
import numpy as np
import numpy.typing as npt
import scipy as sp
import astropy as ap
import astropy.constants


YR = ap.units.year.to(ap.units.s)

def pta_freqs(dur=16.03*YR, num=40, cad=None):
    fmin = 1.0 / dur
    if cad is not None:
        num = dur / (2.0 * cad)
        num = int(np.floor(num))

    cents = np.arange(1, num+2) * fmin

    edges = cents - fmin / 2.0
    cents = cents[:-1]
    return cents, edges

