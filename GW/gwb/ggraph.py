import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

import utils
from utils import YR

FIGSIZE = 6
FONTSIZE = 13
GOLDEN_RATIO = (np.sqrt(5) - 1) / 2

mpl.style.use('default')   # avoid dark backgrounds from dark theme vscode
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.handlelength"] = 1.5
plt.rcParams["lines.solid_capstyle"] = 'round'
# plt.rcParams["font.size"] = FONTSIZE
# plt.rcParams["legend.fontsize"] = FONTSIZE*0.8
# mpl.rcParams['xtick.labelsize'] = FONTSIZE*0.8
# mpl.rcParams['ytick.labelsize'] = FONTSIZE*0.8

LABEL_GW_FREQUENCY_YR = r"GW Frequency $[\mathrm{yr}^{-1}]$"
LABEL_GW_FREQUENCY_HZ = r"GW Frequency $[\mathrm{Hz}]$"
LABEL_GW_FREQUENCY_NHZ = r"GW Frequency $[\mathrm{nHz}]$"
LABEL_SEPARATION_PC = r"Binary Separation $[\mathrm{pc}]$"
LABEL_CHARACTERISTIC_STRAIN = r"GW Characteristic Strain"
LABEL_HARDENING_TIME = r"Hardening Time $[\mathrm{Gyr}]$"
LABEL_CLC0 = r"$C_\ell / C_0$"

PARAM_KEYS = {
    'hard_time': r"phenom $\tau_f$",
    'hard_gamma_inner': r"phenom $\nu_\mathrm{inner}$",
    'hard_gamma_outer': r"phenom $\nu_\mathrm{outer}$",
    'hard_gamma_rot' : r"phenom $\nu_{\mathrm{rot}}$",
    'gsmf_phi0': r"GSMF $\psi_0$",
    'gsmf_mchar0_log10': r"GSMF $m_{\psi,0}$",
    'gsmf_alpha0': r"GSMF $\alpha_{\psi,0}$",
    'gpf_zbeta': r"GPF $\beta_{p,z}$",
    'gpf_qgamma': r"GPF $\gamma_{p,0}$",
    'gmt_norm': r"GMT $T_0$",
    'gmt_zbeta': r"GMT $\beta_{t,z}$",
    'mmb_mamp_log10': r"MMB $\mu$",
    'mmb_plaw': r"MMB $\alpha_{\mu}$",
    'mmb_scatter_dex': r"MMB $\epsilon_{\mu}$",
}

LABEL_DPRATIO = r"$\langle N_\mathrm{SS} \rangle / \mathrm{DP}_\mathrm{BG}$"
LABEL_EVSS = r"$\langle N_\mathrm{SS} \rangle$"
LABEL_DPBG = r"$\mathrm{DP}_\mathrm{BG}$"

COLORS_MPL = plt.rcParams['axes.prop_cycle'].by_key()['color']

def figax(figsize=[7, 5], ncols=1, nrows=1, sharex=False, sharey=False, squeeze=True,
          scale=None, xscale='log', xlabel='', xlim=None, yscale='log', ylabel='', ylim=None,
          left=None, bottom=None, right=None, top=None, hspace=None, wspace=None,
          widths=None, heights=None, grid=True, **kwargs):
    if scale is not None:
        xscale = scale
        yscale = scale

    scales = [xscale, yscale]
    for ii in range(2):
        if scales[ii].startswith('lin'):
            scales[ii] = 'linear'

    xscale, yscale = scales

    if (widths is not None) or (heights is not None):
        gridspec_kw = dict()
        if widths is not None:
            gridspec_kw['width_ratios'] = widths
        if heights is not None:
            gridspec_kw['height_ratios'] = heights
        kwargs['gridspec_kw'] = gridspec_kw

    fig, axes = plt.subplots(figsize=figsize, squeeze=False, ncols=ncols, nrows=nrows,
                             sharex=sharex, sharey=sharey, **kwargs)

    plt.subplots_adjust(
        left=left, bottom=bottom, right=right, top=top, hspace=hspace, wspace=wspace)

    if ylim is not None:
        shape = (nrows, ncols, 2)
        if np.shape(ylim) == (2,):
            ylim = np.array(ylim)[np.newaxis, np.newaxis, :]
    else:
        shape = (nrows, ncols,)

    ylim = np.broadcast_to(ylim, shape)

    if xlim is not None:
        shape = (nrows, ncols, 2)
        if np.shape(xlim) == (2,):
            xlim = np.array(xlim)[np.newaxis, np.newaxis, :]
    else:
        shape = (nrows, ncols)

    xlim = np.broadcast_to(xlim, shape)
    _, xscale, xlabel = np.broadcast_arrays(axes, xscale, xlabel)
    _, yscale, ylabel = np.broadcast_arrays(axes, yscale, ylabel)

    for idx, ax in np.ndenumerate(axes):
        ax.set(xscale=xscale[idx], xlabel=xlabel[idx], yscale=yscale[idx], ylabel=ylabel[idx])
        if xlim[idx] is not None:
            ax.set_xlim(xlim[idx])
        if ylim[idx] is not None:
            ax.set_ylim(ylim[idx])

        if grid is True:
            ax.set_axisbelow(True)
            # ax.grid(True, which='major', axis='both', c='0.6', zorder=2, alpha=0.4)
            # ax.grid(True, which='minor', axis='both', c='0.8', zorder=2, alpha=0.4)
            # ax.grid(True, which='major', axis='both', c='0.6', zorder=2, alpha=0.4)
            # ax.grid(True, which='minor', axis='both', c='0.8', zorder=2, alpha=0.4)

    if squeeze:
        axes = np.squeeze(axes)
        if np.ndim(axes) == 0:
            axes = axes[()]

    return fig, axes

def _twin_hz(ax, nano=True, fs=10, **kw):
    tw = ax.twiny()
    tw.grid(False)
    xlim = np.array(ax.get_xlim()) / YR
    if nano:
        label = LABEL_GW_FREQUENCY_NHZ
        xlim *= 1e9
    else:
        label = LABEL_GW_FREQUENCY_HZ

    tw.set(xlim=xlim, xscale=ax.get_xscale())
    tw.set_xlabel(label, fontsize=fs, **kw)
    return tw
