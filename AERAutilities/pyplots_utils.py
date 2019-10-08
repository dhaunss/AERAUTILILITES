import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator, DictFormatter

import numpy as np
import copy


# Defaults
plt.rc('axes', labelsize=45)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=40)    # fontsize of the tick labels
plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
plt.rc('legend', fontsize=45)    # legend fontsize
plt.rc('figure', titlesize=45)
plt.rc('figure', figsize=(16, 10))
plt.rc('axes', titlesize=45)
plt.rc('legend', numpoints=1)

# this is an proxy for a empty legend entry
def empty_legend(label, marker="None", color="None", markersize=1):
    # yerr to use marker size
    return {"x": np.nan, "y": np.nan, "yerr": np.nan, "marker": marker, "color": color, "markersize": markersize, "label": label}


def cbi_pos():
    divider = make_axes_locatable(plt.gca())
    return {"cax": divider.append_axes("right", "3%", pad="1.5%")}


def save_fig(fig, fname="plot.png", kwargs={}):
    # tight_layout can raise warnings
    print("Save", fname)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", **kwargs)


def add_correlation(var_1, var_2, ax=None, fig=None, posx=None, posy=None):
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(var_1, var_2)

    if ax is None:
        posx, posy = posx or 0.6, posy or 0.15
        plt.text(
                posx, posy, "correlation: %.3f" % (correlation), transform=fig.transFigure, fontsize=20,
                bbox=dict(facecolor='none', edgecolor='k', pad=10.0))

    else:
        posx, posy = posx or 0.8, posy or 0.10
        ax.text(
                posx, posy, "correlation: %.3f" % (correlation), transform=ax.transAxes, fontsize=20,
                bbox=dict(facecolor='none', edgecolor='k', pad=10.0))


def add_func_and_lines(ax, hline=None, vline=None, func=None):

    if hline is not None:
        if not isinstance(hline, list):
            hline = [hline]

        for ele in hline:
            if not isinstance(ele, dict):
                ele = {"y": ele}

            hline_kwargs = {"linestyle": "--", "color": "k", "alpha": 0.7}
            hline_kwargs.update(ele)
            ax.axhline(**hline_kwargs)

    if vline is not None:
        if not isinstance(vline, list):
            vline = [vline]

        for ele in vline:
            if not isinstance(ele, dict):
                ele = {"x": ele}

            vline_kwargs = {"linestyle": "--", "color": "k", "alpha": 0.7}
            vline_kwargs.update(ele)
            ax.axvline(**vline_kwargs)

    if func is not None:
        if not isinstance(func, list):
            func = [func]

        for i, f in enumerate(func):
            func_kwargs = {"linestyle": "-", "color": "C%d" % i}  # defaults , use color that they might match the scattered data
            # geht leider nicht schoener
            func_kwargs.update(f)
            func_kwargs.pop('x', None)
            func_kwargs.pop('y', None)
            if "fmt" in func_kwargs:
                func_kwargs.pop('fmt', None)  # weil nicht unterstuetzt
                print('"fmt" wurde geloescht da nicht unterstuetzt')

            ax.plot(f["x"], f["y"], **func_kwargs)
            ax.legend()

    return ax


def add_secound_y_axis(ax2, ax, sec_axis_dict, x, y, scatter_kwargs={}):
    if "norm" in sec_axis_dict:
        y2 = y / sec_axis_dict["norm"]
        ylim2 = ax.get_ylim() / sec_axis_dict["norm"]
    else:
        y2 = sec_axis_dict["y"]
        ylim2 = sec_axis_dict["lim"] if "lim" in sec_axis_dict else None

    ax2.scatter(x, y2, **scatter_kwargs)
    ax2.set_ylim(ylim2)
    ax2.set_ylabel(sec_axis_dict["label"])


def _scatter(
        ax, var, var_2=None,
        xscale="linear", yscale="linear",
        xlim=None, ylim=None,
        legend={}, grid=False,
        xlabel="", ylabel="", title="",
        xticks={}, yticks={},
        scatter_kwargs={}):

    # allows to give function simple x, y
    if var_2 is not None:
        var = {"x": var, "y": var_2}

    if not isinstance(var, list):
        var = [var]

    # scatter_kwargs is for additional arguments
    for idx, v in enumerate(var):
        kwargs = copy.copy(scatter_kwargs)
        kwargs.update(v)
        if "xerr" in kwargs or "yerr" in kwargs or "linestyle" in kwargs:
            sct = ax.errorbar(**kwargs)
        else:
            # with python 2 only one dict can be passed with **
            sct = ax.scatter(**kwargs)

    # Value Erreor was raised when changing order of set scale and lim.... fuck pyplot
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # needs to be before secound_y_axis
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(title)

    if len(xticks) > 0:
        ax.set_xticks(**xticks)
    if len(yticks) > 0:
        ax.set_yticks(**yticks)

    # set legend if not none
    leg = ax.legend(**legend) if legend is not None else None

    if not isinstance(grid, list):
        grid = [grid]

    for g in grid:
        if not isinstance(g, dict):
            g = {"b": g}
        ax.grid(**g)

    return ax, sct, leg


def setup_axes3(fig, rect, component, fig_title=None, title_size=25, inclined=False, ylim=None):
    """
    Sometimes, things like axis_direction need to be adjusted.
    """

    # Angle in degree
    angle_ticks = [(0., r"$0^\circ$"),
                   (15., r"$15^\circ$"),
                   (30., r"$30^\circ$"),
                   (45., r"$45^\circ$"),
                   (60., r"$60^\circ$"),
                   (75., r"$75^\circ$"),
                   (90., r"$90^\circ$")]

    if not inclined:
        # rotate a bit for better orientation
        tr_rotate = Affine2D().translate(90, 0)

        # scale degree to radians
        tr_scale = Affine2D().scale(np.pi / 180., 1.)

        # ploting zenith angle range
        ra0, ra1 = 0., 100.

        grid_locator1 = FixedLocator([v for v, s in angle_ticks])
        tick_formatter1 = DictFormatter(dict(angle_ticks))

    else:
        # rotate a bit for better orientation
        tr_rotate = Affine2D().translate(-5, 0)

        # scale degree to radians
        tr_scale = Affine2D().scale(np.pi / 90., 1.)

        # ploting zenith angle range
        ra0, ra1 = 50., 100.

        grid_locator1 = None
        tick_formatter1 = DictFormatter(dict(angle_ticks))

    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    # Angle in minutes
    # grid_locator1 = angle_helper.LocatorHMS(6)
    # tick_formatter1 = angle_helper.FormatterHMS()

    grid_locator2 = MaxNLocator(11)

    if ylim is not None:
        cz0, cz1 = ylim
    else:
        cz0, cz1 = 0, 50.

    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                                      extremes=(ra0, ra1, cz0, cz1),
                                                      grid_locator1=grid_locator1,
                                                      grid_locator2=grid_locator2,
                                                      tick_formatter1=tick_formatter1,
                                                      tick_formatter2=None,
                                                      )

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    if fig_title is not None:
        plt.title(fig_title, fontsize=title_size, loc="left")

    # adjust axis
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")

    ax1.axis["bottom"].set_visible(False)
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")

    if component == "horizontal" or component == "hori":
        ax1.axis["left"].label.set_text(r"$|H_{\phi}|$ [m]")
    elif component == "meridional":
        ax1.axis["left"].label.set_text(r"$|H_{\theta}|$ [m]")
    elif component == "vertical-horizontal":
        ax1.axis["left"].label.set_text(r"$|H_{\theta,hor}|$ [m]")
    elif component == "vertical-vertical":
        ax1.axis["left"].label.set_text(r"$|H_{\theta.vert}|$ [m]")
    elif component == "vertical":
        ax1.axis["left"].label.set_text(r"$|H_{v}| = |H_{\theta}| \cdot \sin(\theta)$ [m]")

    ax1.axis["left"].label.set_fontsize(24)
    ax1.axis["left"].major_ticklabels.set_fontsize(22)
    ax1.axis["top"].label.set_text(r"$\Theta$")
    ax1.axis["top"].label.set_fontsize(24)
    ax1.axis["top"].major_ticklabels.set_fontsize(22)

    ax1.grid(True)

    # create a parasite axes whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch  # for aux_ax to have a clip path as in ax
    ax1.patch.zorder = 0.9  # but this has a side effect that the patch is
    # drawn twice, and possibly over some other
    # artists. So, we decrease the zorder a bit to
    # prevent this.

    return ax1, aux_ax
