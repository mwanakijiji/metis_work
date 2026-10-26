"""
Warnings for results that are provisional until more input is available, written to the
logs and stamped on the plots.

IMG-OPT-01 (FoV) and IMG-OPT-02 (distortion) need more distortion information from DD
before their simulations and analyses can be relied on.
"""

import logging

from matplotlib.figure import Figure

TBD_DISTORTION_MSG = "TBD MORE DISTORTION INFO FROM DD NEEDED FIRST"


def log_tbd_distortion(context=""):
    """Log a prominent warning that distortion info from DD is still needed."""
    line = "!" * 72
    logging.warning(line)
    logging.warning("!!! %s%s", TBD_DISTORTION_MSG, f" ({context})" if context else "")
    logging.warning(line)


def add_tbd_banner(fig, message=TBD_DISTORTION_MSG):
    """Stamp a red warning banner across the top of a matplotlib figure (once)."""
    if getattr(fig, "_tbd_banner", False):
        return fig
    fig.text(
        0.5,
        0.995,
        message,
        ha="center",
        va="top",
        color="red",
        fontsize=14,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "red", "alpha": 0.85},
        zorder=1000,
    )
    fig._tbd_banner = True  # pylint: disable=protected-access
    return fig


def stamp_all_saved_figures(message=TBD_DISTORTION_MSG):
    """
    From now on in this process, stamp the warning banner on every matplotlib figure
    before it is saved (Figure.savefig, which plt.savefig also calls). For scripts whose
    plots are made deep in shared modules.
    """
    if getattr(Figure.savefig, "_tbd_wrapped", False):
        return
    original_savefig = Figure.savefig

    def savefig(self, *args, **kwargs):
        add_tbd_banner(self, message)
        return original_savefig(self, *args, **kwargs)

    savefig._tbd_wrapped = True  # pylint: disable=protected-access
    Figure.savefig = savefig
