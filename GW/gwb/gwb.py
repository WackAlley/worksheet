import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import ggraph as plot
import utils
from utils import YR

fobs, fobs_edges = utils.pta_freqs(num=300)
print(f"Number of frequency bins: {fobs.size-1}")
print(f"  between [{fobs[0]*YR:.2f}, {fobs[-1]*YR:.2f}] 1/yr")
print(f"          [{fobs[0]*1e9:.2f}, {fobs[-1]*1e9:.2f}] nHz")


fig, ax = plot.figax(xlabel='Frequency $f_\mathrm{obs}$ [1/yr]', ylabel='Characteristic Strain $h_c$')

xx = fobs * YR

yy = 1e-15 * np.power(xx, -2.0/3.0)
ax.plot(xx,yy,'k--', alpha=0.25, lw=2.0)


plot._twin_hz(ax, nano=True)
plt.show()
