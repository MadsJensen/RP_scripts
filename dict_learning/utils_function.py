import matplotlib.pyplot as plt
import numpy as np
import mne
from my_settings import erf_raw

epo = mne.read_epochs(erf_raw + "0008_classic_ar_grads_erf_hg-epo.fif")
info = epo.info
sfreq = info['sfreq']
t = epo.times[::2]


def plot_atoms(cdl, plotted_atoms, show=False):

    # preselected atoms of interest
    n_plots = 3  # number of plots by atom
    n_columns = min(6, len(plotted_atoms))
    split = int(np.ceil(len(plotted_atoms) / n_columns))
    figsize = (4 * n_columns, 3 * n_plots * split)
    fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)
    for ii, kk in enumerate(plotted_atoms):

        # Select the axes to display the current atom
        print("\rDisplaying {}-th atom".format(kk), end='', flush=True)
        i_row, i_col = ii // n_columns, ii % n_columns
        it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

        # Select the current atom
        u_k = cdl.u_hat_[kk]
        v_k = cdl.v_hat_[kk]

        # Plot the spatial map of the atom using mne topomap
        ax = next(it_axes)
        mne.viz.plot_topomap(u_k, info, axes=ax, show=False)
        ax.set(title="Spatial pattern %d" % (kk, ))

        # Plot the temporal pattern of the atom
        ax = next(it_axes)
        ax.plot(t, v_k)
        ax.set_xlim(t[0], t[-1])
        ax.set(xlabel='Time (sec)', title="Temporal pattern %d" % kk)

        # Plot the power spectral density (PSD)
        ax = next(it_axes)
        psd = np.abs(np.fft.rfft(v_k, n=256))**2
        frequencies = np.linspace(0, sfreq / 2.0, len(psd))
        ax.semilogy(frequencies, psd, label='PSD', color='k')
        ax.set(xlabel='Frequencies (Hz)',
               title="Power spectral density %d" % kk)
        ax.grid(True)
        ax.set_xlim(0, info['lowpass'])
        # ax.set_ylim(1e-4, 1e2)
        ax.legend()
    print("\rDisplayed {} atoms".format(len(plotted_atoms)).rjust(40))

    if show:
        plt.show()

    return fig
