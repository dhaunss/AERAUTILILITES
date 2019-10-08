import matplotlib as mpl
import matplotlib.pyplot as plt

import AERAutilities.pyplots_utils as putils
from radiotools import plthelpers as php
from radiotools import helper as hep

from RadioAnalysis.utilities import utils as rdutils

import numpy as np

# to use radiotools directly
rphp = php
utils = putils


def save_fig(fig, fname="plot.png", kwargs={}):
    putils.save_fig(fig, fname, kwargs)


def scatter_2d(
        var, var_2=None,
        scatter_kwargs={},
        xscale="linear", yscale="linear",
        xlim=None, ylim=None,
        xlabel="", ylabel="", title="",
        func=None, residual=False, grid=False,
        hline=None, vline=None,
        correlation=False, yaxis2=None,
        legend={}, figsize=None,
        tick_params={"axis": "both"},
        fname="scatter.png"):

    # vline, hline can be eigther dicts or single floats
    # var and var_2 can be eigther x and y or
    # var is a single or list of dicts which have a key for x and y (then var_2 has to be None)
    # func should by a single or lists of dicts with x and y keys
    # yaxis2 can be a dict for a secound y axis: needs eigther "norm" or "y"

    if not residual:
        fig, ax = plt.subplots(1, figsize=figsize)
        fig.subplots_adjust(wspace=0.3, hspace=0, bottom=0.3, left=0.05, right=0.95)  # ueberarbeiten
    else:
        figsize = (16, 12) or figsize  # for others controlled by rc paras
        fig = plt.figure(figsize=figsize)
        # fig, (ax, ax_res) = plt.subplots(2, sharex=True, figsize=figsize)
        fig.subplots_adjust(hspace=0, wspace=0.3, left=0.05, right=0.95)

        ax = fig.add_axes((.1,.3,.85,.6))
        ax_res = fig.add_axes((.1,.1,.85,.19))


    if "titlesize" in legend:
        leg_title = legend["title"]
        titlesize = legend["titlesize"]
        legend.pop("title", None)
        legend.pop("titlesize", None)
    else:
        leg_title, titlesize = "", 0

    ax, sct, leg = putils._scatter(
        ax, var, var_2,
        xscale=xscale, yscale=yscale, grid=grid,
        xlim=xlim, ylim=ylim, legend=legend,
        xlabel=xlabel, ylabel=ylabel, title=title,
        scatter_kwargs=scatter_kwargs)

    leg.set_title(leg_title, prop={'size': titlesize})

    if residual:
        if isinstance(residual, dict):
            var_res = residual['var']
            residual.pop('var', None)

        else:
            var_res = residual
            residual = {}

        # set ylabel offset hardcoded
        if 'yoff' in residual:
            ax.yaxis.set_label_coords(residual['yoff'], 0.5)
            ax_res.yaxis.set_label_coords(residual['yoff'], 0.5)
            residual.pop('yoff', None)

        for t in ['hline', 'vline', 'func']:
            if t in residual:
                ax_res = putils.add_func_and_lines(ax_res, **{t: residual[t]})
                residual.pop(t, None)

        ax_res, sct_res, _ = putils._scatter(
            ax_res, var_res, var_2=None,
            xscale=xscale, xlim=ax.get_xlim(), grid=grid, xlabel=xlabel, legend=None,
            **residual)

        #Remove x-tic labels for the axis frame
        ax.set_xticklabels(ax.get_xticklabels(), alpha=0)

    # add a secound (scaled) y-axis
    # does not work for lists, would it make sense?
    if yaxis2 is not None:
        ax2 = ax.twinx()

        if var_2 is None:
            x, y = var["x"], var["y"]
        else:
            x, y = var, var_2

        putils.add_secound_y_axis(ax2, ax, yaxis2, x, y, scatter_kwargs)

    ax = putils.add_func_and_lines(ax, hline, vline, func)

    if correlation:
        putils.add_correlation(x, y, fig=fig)

    #plt.savefig(fname)
    save_fig(fig, fname)


def scatter_color_2d(
        var, var_2=None, color_var=None,
        xscale="linear", yscale="linear",
        xlim=None, ylim=None,
        xlabel="", ylabel="", title="",
        func=None, grid=False,
        hline=None, vline=None, legend={},
        correlation=False, yaxis2=None,
        cscale="linear", clim=None, clabel="", cmap="viridis", c_kwargs={},
        scatter_kwargs={}, figsize=(16, 10),
        fname="scatter_color.png"):

    # vline, hline can be eigther dicts or single floats

    fig, ax = plt.subplots(1, figsize=figsize)
    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

    norm = mpl.colors.LogNorm() if cscale == "log" else None
    vmin, vmax = clim or (None, None)

    # default setting for scatter color
    scatter_kwargs.update({"cmap": plt.get_cmap(cmap), "norm": norm, "vmin": vmin, "vmax": vmax})
    scatter_kwargs.update(c_kwargs)

    if var_2 is not None:
        scatter_kwargs.update({"c": color_var})

    ax, sct, leg = putils._scatter(
        ax, var, var_2,
        xscale=xscale, yscale=yscale, grid=grid,
        xlim=xlim, ylim=ylim, legend=legend,
        xlabel=xlabel, ylabel=ylabel, title=title,
        scatter_kwargs=scatter_kwargs)

    # cbi_pos() does not work here
    pad = 0.02 if yaxis2 is None else 0.1
    cbi = plt.colorbar(sct, orientation='vertical', shrink=0.99, pad=pad)
    labelsize = mpl.rcParams['axes.labelsize']
    ticksize = mpl.rcParams['ytick.labelsize']
    cbi.ax.tick_params(axis='both', **{"labelsize": ticksize})
    cbi.set_label(clabel, fontsize=labelsize)

    ax = putils.add_func_and_lines(ax, hline, vline, func)

    save_fig(fig, fname)


def from_2d_to_1d(
                data_2d, bins=20, ax=None,
                xlabel="", ylabel="Entries", title="",
                median=True, mean=True,
                kwargs=None, stat_kwargs=None,
                log_kwargs={"loc": "upper center", "fontsize": 20},
                fname="example_2d_to_1d.png"):

    data_1d = np.reshape(data_2d, data_2d.size)

    kwargsx = kwargs or {'facecolor': '0.7', 'alpha': 1, 'edgecolor': "k"}
    stat_kwargsx = stat_kwargs or {"ha": "right", "fontsize": 20, "posx": 0.98, "posy": 0.98}

    if ax is None:
        fig, ax1 = php.get_histogram(
            data_1d, bins, xlabel=xlabel,
            ylabel=ylabel, title=title, kwargs=kwargsx,
            stat_kwargs=stat_kwargsx)

    else:
        php.get_histogram(
            data_1d, bins, xlabel=xlabel, ax=ax,
            ylabel=ylabel, title=title, kwargs=kwargsx,
            stat_kwargs=stat_kwargsx)

    if mean:
        plt.axvline(data_1d.mean(), linestyle='dashed', linewidth=2, color=kwargsx["edgecolor"], label="Mean")

    if median:
        plt.axvline(np.median(data_1d, axis=0), linestyle='dotted', linewidth=2, color=kwargsx["edgecolor"], label="Median")

    ax1.tick_params(axis='both', **{"labelsize": 20})
    plt.legend(**log_kwargs)

    if ax is None:
        save_fig(fig, fname)
    else:
        return ax


def footprint(station_pos, energy_fluence, xx=None, yy=None, xlim=None, ylim=None, figsize=(14, 12),
              title='', vmin=None, vmax=None, with_station=True, cmap='gnuplot2_r', cbi_kwargs={'shrink': 0.8},
              ratio=None, clabel=r'$f$ / $\frac{eV}{m^2}$', fname='propaganda.png'):
    from scipy import interpolate as intp

    dat_station = {"x": station_pos[:, 0], "y": station_pos[:, 1], "c": energy_fluence, 'edgecolor': 'k', 's': 200}
    interp_func = intp.Rbf(station_pos[:, 0], station_pos[:, 1], energy_fluence, smooth=0, function='quintic')

    if xx is None and yy is None:
        xs = np.linspace(np.amin(station_pos[:, 0]), np.amax(station_pos[:, 0]), 100)
        ys = np.linspace(np.amin(station_pos[:, 1]), np.amax(station_pos[:, 1]), 100)
        xx, yy = np.meshgrid(xs, ys)

    fp_interp = interp_func(xx, yy)

    fig = plt.figure(figsize=figsize)  # default is (8,6)
    ax = fig.add_subplot(111, aspect='equal')

    ax, im = histogram2d(xx, yy, fp_interp, ax1=ax,
                                 title=title, xlabel=r'v$\,$x$\,$B / m', ylabel=r'v$\,$x$\,v$x$\,$B / m',
                                 colorbar=False, shading='gouraud', cmap=cmap, clim=(vmin, vmax))

    if with_station:
        ax, sct, leg = putils._scatter(ax, dat_station, grid=True,
                                               title=title, xlabel=r'v$\,$x$\,$B / m', ylabel=r'v$\,$x$\,v$x$\,$B / m',
                                               scatter_kwargs={'cmap': cmap, "vmin": vmin, "vmax": vmax}, legend=None)
    ax.grid(True)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if ratio is not None:
        ax.set_aspect(aspect=1/ratio)

    cbi = plt.colorbar(im, **cbi_kwargs)
    cbi.ax.tick_params(axis='both', **{"labelsize": 30})
    cbi.set_label(clabel, fontsize=35)
    save_fig(fig, fname)


def histogram2d(x=None, y=None, z=None,
                bins=10, range=None, clim=(None, None),
                xscale="linear", yscale="linear", cscale="linear",
                normed=False, cmap=None,
                hline=None, vline=None, func=None,
                ax1=None, grid=True, shading='flat', colorbar=True,
                profile=False, cbi_kwargs={'orientation': 'vertical'},
                xlabel="", ylabel="", clabel="", title="",
                fname="hist2d.png"):

    if z is None and (x is None or y is None):
        sys.exit("z and (x or y) are all None")

    if ax1 is None:
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
    else:
        ax = ax1

    if z is None:
        z, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range)
        z = z.T
    else:
        xedges, yedges = x, y

    color_norm = None
    if cscale == "log":
        color_norm = mpl.colors.LogNorm()

    if normed:
        if normed == "colum":
            z = z / np.sum(z, axis=0)
        elif normed == "row":
            z = z / np.sum(z, axis=1)[:, None]
        elif normed == "colum1":
            z = z / np.amax(z, axis=0)
        elif normed == "row1":
            z = z / np.amax(z, axis=1)[:, None]
        else:
            sys.exit("Normalisation %s is not known.")

    vmin, vmax = clim
    im = ax.pcolormesh(xedges, yedges, z, shading=shading, vmin=vmin, vmax=vmax , norm=color_norm, cmap=cmap)

    if profile:
        xbins = bins[0] if isinstance(bins, list) else bins
        hist, x, y, yerr = rdutils.get_binned_data(x, y, xbins)
        ax.errorbar(x=x[hist > 5], y=y[hist > 5], yerr=yerr[hist > 5], marker='o', color='r', linestyle='')

    if colorbar:
        cbi = plt.colorbar(im, **cbi_kwargs, **putils.cbi_pos())
        cbi.ax.tick_params(axis='both', **{"labelsize": 30})
        cbi.set_label(clabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.set_title(title)

    ax = utils.add_func_and_lines(ax, hline=hline, vline=vline, func=func)

    if ax1 is None:
        save_fig(fig, fname)
    else:
        return ax, im

def polar_hist(
        phi, zenith, var=None, norm=True,
        std=False,
        phi_bin=16, zenith_bin=4,
        ax1=None, ax_loc=[0.01, 0.1, 0.8, 0.8],
        alpha=1, clabel="Entries", cscale="lin", cmap=None, clim=None,
        title="",
        fname="direction_hist.png"):

    if ax1 is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_axes(ax_loc, projection='polar')
    else:
        ax = ax1

    if isinstance(zenith_bin, int):
        zenith_edges = np.linspace(0, 85, zenith_bin + 1)
    else:
        zenith_edges = zenith_bin

    if isinstance(phi_bin, int):
        phi_edges = np.linspace(0, 2*np.pi, phi_bin + 1)
    else:
        phi_edges = phi_bin

    n, _, _ = np.histogram2d(np.rad2deg(zenith), phi, [zenith_edges, phi_edges])
    n_mask = n != 0

    if var is not None:
        h, _, _ = np.histogram2d(np.rad2deg(zenith), phi, [zenith_edges, phi_edges], weights=var, normed=False)  # , bins=[16, 18], range=[[0, 2*np.pi], [0, np.pi]])
        if norm or std:
            h[n_mask] = h[n_mask] / n[n_mask]
            clabel = "mean" if clabel == "Entries" else clabel
            if std:
                h2, _, _ = np.histogram2d(np.rad2deg(zenith), phi, [zenith_edges, phi_edges], weights=var ** 2, normed=False)
                h2[n_mask] = h2[n_mask] / n[n_mask]
                h2[n_mask] = np.abs(h2[n_mask] - np.power(h[n_mask], 2.))
                h = np.sqrt(h2)
                clabel = "std" if clabel == "Entries" else clabel
        else:
            clabel = "var" if clabel == "Entries" else clabel

    else:
        h = n

    color_norm = mpl.colors.LogNorm() if cscale == "log" else None

    fig, ax = plt.subplots(ncols=1, subplot_kw=dict(projection='polar'), figsize=(10, 10))
    p, z = np.meshgrid(phi_edges, zenith_edges)

    if clim is not None:
        vmin, vmax = clim
    else:
        vmin, vmax = None, None

    im = ax.pcolormesh(p, z, h, cmap=cmap, norm=color_norm, vmin=vmin, vmax=vmax)
    cbi = plt.colorbar(im, orientation='vertical', shrink=0.9)
    cbi.ax.tick_params(axis='both', **{"labelsize": 20})
    cbi.set_label(clabel, fontsize=25)

    # plot mag field
    mag_azi = hep.get_magneticfield_azimuth(hep.get_declination(hep.get_magnetic_field_vector("auger")))
    mag_zen = hep.get_magneticfield_zenith(hep.get_inclination(hep.get_magnetic_field_vector("auger")))
    ax.scatter(mag_azi, np.rad2deg(mag_zen), marker="*", s=300, color="r")

    # FuncFormatter can be used as a decorator
    @mpl.ticker.FuncFormatter
    def rad_formatter(x, pos):
        if x < np.amin(zenith_edges):
            return ""
        if x % 20 != 0:
            return ""
        return r"%.0f$^\circ$" % x

    ax.yaxis.set_major_formatter(rad_formatter)

    ax.grid(True)

    if ax1 is None:
        save_fig(fig, fname=fname)


def polar_bar(
        phi, var, var_range=[0, 100],
        ax=None, ax_loc=[0.01, 0.1, 0.8, 0.8],
        color=None, alpha=1, norm=0, clabel="",
        title="Number of Events per arrival direction",
        fname="direction_bar.png"):

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_axes(ax_loc, projection='polar')
    else:
        ax1 = ax

    heatmap, xedges, yedges = np.histogram2d(phi, var, bins=[8, 1], range=[[0, 2 * np.pi], var_range])

    bin_center = xedges[1:] - (xedges[1] - xedges[0]) / 2
    width = (xedges[1] - xedges[0]) * 0.8
    N_azimuth = np.squeeze(heatmap)

    if norm:
        N_azimuth = N_azimuth * norm

    bars = ax1.bar(bin_center, N_azimuth, width=width, bottom=0.0, label=label, alpha=alpha)

    ax1.set(title=title)

    for bar in bars:
        bar.set_color(color)
        bar.set_alpha(alpha)

    if ax is None:
        save_fig(fig, fname)
    else:
        return ax1


def polar_scatter(
        phi, var,
        ax=None, ax_loc=[0.01, 0.1, 0.8, 0.8],
        color=None, ylim=None, label=None,
        ytitle=r"$^\circ$zenith", title="Arrival Directions",
        fname="polar_example.png"):

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_axes(ax_loc, projection='polar')  # axisbg='#d5de9c')
    else:
        ax1 = ax

    ax, sct, leg = putils._scatter(
        ax, phi, var,
        title=title, ylim=ylim, grid=True,
        scatter_kwargs={"marker": "o", "color": color, "label": label})

    # Lucky punch
    ypos = np.amax(var) + (np.amax(var)) / 10. + 5
    ax1.text(np.deg2rad(22), ypos, ytitle, fontsize=20)

    if ax is None:
        save_fig(fig, fname)
    else:
        return ax1


def ldf_1d(
        xvar, y=None, xerr=0, yerr=0,
        func=None, hline=None, vline=None,
        kwargs={"linestyle": "", "marker": "o"},
        xscale="linear", yscale="linear",
        xlim=None, ylim=None,
        legend={},
        xlabel="off axis distance r / m", ylabel=r"energy fluence f / $\frac{eV}{m^2}$", title="",
        grid=False, figsize=(16, 10), yaxis=None,
        fname="1d_ldf.png"):

    # allows to give function simple x, y
    if y is not None:
        xvar = {"x": xvar, "y": y, "xerr": xerr, "yerr": yerr}
        y = None

    fig, ax = plt.subplots(1, figsize=figsize)
    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

    ax, sct, leg = putils._scatter(
        ax, xvar, y,
        xscale=xscale, yscale=yscale, grid=grid,
        xlim=xlim, ylim=ylim, legend=legend,
        xlabel=xlabel, ylabel=ylabel, title=title,
        scatter_kwargs=kwargs)

    ax = putils.add_func_and_lines(ax, hline, vline, func)

    if yaxis is not None:
        # set only axis with new label
        if isinstance(yaxis, str):
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_ylabel(yaxis)

    save_fig(fig, fname)


def station_map(
            data,
            clabel="",
            vb_vvb=False,
            title="",
            xlim=None, ylim=None, cmap="viridis",
            clim=None, alpha=1, cscale="linear",
            fname="station_map.png"):

    if not isinstance(data, list):
        data = [data]

    xlabel = r"v x B [m]" if vb_vvb else "x [m]"
    ylabel = r"v x v x B [m]" if vb_vvb else "y [m]"
    if title == "":
        title = "Station Map in Shower Coordinates" if vb_vvb else "Station Map in Cartesian Coordinates"

    norm = mpl.colors.LogNorm() if cscale == "log" else None
    vmin, vmax = clim or (None, None)

    fig = plt.figure(figsize=(14, 12))  # default is (8,6)
    ax = fig.add_subplot(111, aspect='equal')

    ax, sct, leg = putils._scatter(
        ax, data, grid=True, xlim=xlim, ylim=ylim,
        xlabel=xlabel, ylabel=ylabel, title=title, scatter_kwargs={"cmap": plt.get_cmap(cmap), "norm": norm, "vmin": vmin, "vmax": vmax})

    if "c" in data[0]:
        # cbi_pos() does not work here
        pad = 0.02
        cbi = plt.colorbar(sct, orientation='vertical', shrink=0.9, pad=pad)
        cbi.ax.tick_params(axis='both', **{"labelsize": 20})
        cbi.set_label(clabel, fontsize=25)

    ax.axis("equal")
    ax.set_aspect("equal")
    save_fig(fig, fname)
