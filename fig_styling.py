## script to apply extra styling to figures shared across multiple scripts
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator


def style_axis(ax, major_len=10, major_w=2, minor_len=5, minor_w=1.5,
            pad=10, label_weight='bold', fontsize=20):
    # Pad for major ticks
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(pad)

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3)

    # --- X axis minor ticks ---
    if ax.get_xscale() == 'log':
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    # --- Y axis minor ticks ---
    if ax.get_yscale() == 'log':
        ymin, ymax = ax.get_ylim()
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10),numticks=11))
        # numticks=np.log10(ymax/ymin)*9)
        print("ticks, ", np.log10(ymax/ymin)*9)

    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Major ticks
    ax.tick_params(axis='both', which='major',
                top=True, right=True,
                bottom=True, left=True,
                length=major_len, width=major_w, pad=pad, labelsize=fontsize*1.05)

    # Minor ticks
    ax.tick_params(axis='both', which='minor',
                top=True, right=True,
                bottom=True, left=True,
                length=minor_len, width=minor_w)

    # Label weight
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight(label_weight)


        