import numpy as np
import astropy as ap
import astropy.constants
import matplotlib.pyplot as plt


YR = ap.units.year.to(ap.units.s)

def freqs(dur=16.03*YR, num=40, cad=None):
    fmin = 1.0 / dur
    if cad is not None:
        num = dur / (2.0 * cad)
        num = int(np.floor(num))

    cents = np.arange(1, num+2) * fmin

    edges = cents - fmin / 2.0
    cents = cents[:-1]
    return cents, edges
    
def b1(f):
    return 1.3e-11 * (1e-4 / f)


OBS_DUR = 10.0 * YR
NUM_FREQS = 20
fobs, fobs_edges = freqs(dur=OBS_DUR, num=NUM_FREQS)


plt.loglog(fobs*1e9, b1(fobs))
plt.show()

plt.loglog(fobs,fobs**6)
plt.loglog(fobs, fobs**(2/3))
plt.show()
