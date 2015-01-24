from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
import warnings
# from mpl_toolkits.axes_grid1 import ImageGrid


def subplot_stack(x, Y, fig=None, ncols=None, nrows=None, title='',
                  colors=['k', 'b'], hspace=0, wspace=0, use_yticks=False,
                  use_xticks=True, enumerate_subplots=False, ytick_bins=None,
                  xtick_bins=None, ylabels=None, xlabels=None, legends=None,
                  share_ylim=False, figsize=None):
    """ Stack a series of 1D plots

    support for calls to an existing fig is still experimental.  should work
    in case where x is the same both times and Y has the same dimensions each
    time
        
    Parameters
    ----------
    x : array_like
        x-axis array.  Must either be a 1D array to be shared among all
        subplots, or it must match the size of Y.
    Y : array_like
        2D array of data to plot.  Separate subplots will correspond to the
        second dimension of the array.
    fig : matplotlib.figure.Figure, optional
        if provided, use the existing figure instead of creating a new one
    ncols : int, optional
        Number of columns to distribute the subplots over (default=1).  Number
        of rows will be inferred.
    nrows : int, optional
        Number of rows to distribute the subplots over.  The number of columns
        will be inferred
    title : str, optional
        title for the figure
    colors : str or list of str, optional
        first and second colors in list will be applied to the real and
        imagingary components of each subplot, respectively.
    hspace : float, optional
        amount of height reserved for white space between subplots
    wspace : float, optional
        amount of width reserved for white space between subplots
    use_xticks : bool, optional
        if False, suppress xticks and xtick_labels
    use_yticks : bool, optional
        if False, suppress yticks and ytick_labels
    enumerate_subplots : bool, optional
        if True, apply ylabels to enumerate the subplots. equivalent to
        setting ylabels = [str(d) for d in range(Y.shape[1])]
    ytick_bins : int, optional
        number of tick mark bins for the y axis
    xtick_bins : int, optional
        number of tick mark bins for the x axis
    xlabels : list of str, optional
        list of x axis labels
    ylabels : list of str, optional
        list of y axis labels
    legends : list of str, optional
        list of legend labels
    share_ylim : bool, optional
        If True, all subplots will have the same y axis scaling, otherwise
        each is scaled independently
    figsize : tuple, optional
        figure size in inches: (width, height).
        
        
    """
    x = np.asanyarray(x).squeeze()
    Y = np.asanyarray(Y).squeeze()

    # TODO: remove this auto-reshaping and raise warning instead?
    if x.ndim > 1 and x.shape[1] > x.shape[0]:
        x = x.T
    if Y.ndim > 1 and Y.shape[1] > Y.shape[0]:
        Y = Y.T

    if x.ndim > 1:
        if (x.shape[1] != 1) and (x.shape[1] != Y.shape[1]):
            raise ValueError("x must either match the shape of Y or " +
                             "have only one non-singleton dimension")

        if x.shape[1] == 1:
            single_x = True
            unequal_x = False
        else:
            single_x = False
        
    # if columns off x are different can't share the same x-axis
            unequal_x = np.sum(np.diff(x, axis=1)) > 0

            if use_xticks and unequal_x:
                # make sure there is room for the x-ticks
                hspace = max(hspace, 0.2)
    else:
        single_x = True
        unequal_x = False

    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.ndim > 2:
        raise ValueError("Y must be 1D or 2D")
    if Y.shape[0] != x.shape[0]:
        raise ValueError("x and Y shapes incompatible")

    if nrows and ncols:
        raise ValueError("specify either nrows or ncols, but not both")
    if ncols:
        nrows = int(ceil(Y.shape[1] / ncols))
    if nrows:
        ncols = int(ceil(Y.shape[1] / nrows))
    else:  # default to ncols=1
        ncols = 1
        nrows = Y.shape[1]

    if isinstance(colors, str):
        colors = [colors, ]
    elif len(colors) > 2:
        raise ValueError(">2 elements provided in colors list")

    if np.iscomplexobj(Y):
        is_complex = True
        # if only 1 color specified, use it for the imaginary axis too
        if len(colors) == 1:
            colors = [colors[0], ] * 2
    else:
        is_complex = False

    middle_col = int(floor(ncols / 2))

    if not fig:
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()            
        axes_list = []
        use_existing_fig = False
    else:
        use_existing_fig = True
        if figsize is not None:
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])
        axes_list = fig.axes

    if share_ylim:
        if is_complex:
            ymin = min(Y.real.min(), Y.imag.min())
            ymax = max(Y.real.min(), Y.imag.max())
        else:
            ymin = Y.min()
            ymax = Y.max()
        # set ylim slightly wider than (ymin, ymax)
        ylim_pad = 0.05*(ymax-ymin)
        ylim = (ymin-ylim_pad, ymax+ylim_pad)
        
    cnt = 0
    for r in range(nrows):
        for c in range(ncols):

            if single_x:
                x_current = x
            else:
                x_current = x[:, cnt]

            # axes_list.append(plt.subplot2grid((nrows,ncols),(r,c),colspan=1,rowspan=1,sharex=axes_list[0]))
            if not use_existing_fig:
                axes_list.append(
                    plt.subplot2grid(
                        (nrows, ncols), (r, c), colspan=1, rowspan=1))

            # get current axis
            ax = axes_list[cnt]

            if use_existing_fig:
                ax.hold(True)

            if is_complex:
                ax.plot(x_current, Y[:, cnt].real, colors[0],
                        x, Y[:, cnt].imag, colors[1])
            else:
                ax.plot(x_current, Y[:, cnt], colors[0])

            if share_ylim:
                ax.set_ylim(ylim)
                
            if enumerate_subplots:  # number the subplots
                ax.set_ylabel('{}   '.format(cnt), rotation='horizontal')
                # ax.set_ylabel('Y[:,{}]'.format(cnt),rotation='vertical')

            if ylabels and (len(ylabels) >= cnt):
                if enumerate_subplots:
                    warnings.warn(
                        "enumerate_subplots option overridden by ylabels")
                ax.ylabel = ylabels[cnt]
            if xlabels and (len(xlabels) >= cnt):
                ax.xlabel = xlabels[cnt]
            if legends and (len(legends) >= cnt):
                ax.legend((legends[cnt],))

            cnt += 1

            if use_xticks:
                if xtick_bins:
                    ax.locator_params(axis='x', nbins=xtick_bins)
                # help avoid y-ticks overlap by hiding the bottom tick
                if ncols > 1 and wspace < 0.25:
                    ax.set_xticks(ax.get_xticks()[1:])
            # remove xticks from all but bottommost
            if (r < (nrows - 1) and (not unequal_x)) or (not use_xticks):
                plt.setp(ax.get_xticklabels(), visible=False)

            if use_yticks:
                if ytick_bins:
                    ax.locator_params(axis='y', nbins=ytick_bins)
                # help avoid y-ticks overlap by hiding the bottom tick
                if nrows > 1 and hspace < 0.25:
                    ax.set_yticks(ax.get_yticks()[1:])
                if c > 0:  # remove y-ticks from all but right-most axis
                    plt.setp(ax.get_yticklabels(), visible=False)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(ax.get_yticklines(), visible=False)

            if title and r == 0 and c == middle_col:
                ax.set_title(title)

    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    fig.show()
    return fig
