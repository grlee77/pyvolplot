# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import inspect
import warnings
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from .montager import montager, montager4d, add_lines, _calc_rows
from .centerplanes_stack import centerplanes_stack, centerplanes_stack_RGB
from .mips import calc_mips
from .utils import add_inner_title
from matplotlib import is_string_like


def masked_overlay(image, overlay_cmap=plt.cm.hot, vmin=None, vmax=None,
                   alpha_image=None, maskval=0):  # , f=None):
    """ overlay another volume via alpha transparency """

    if vmin is None:
        vmin = image.min()
    if vmax is None:
        vmax = image.max()
    if alpha_image is None:
        if maskval is not None:
            alpha_mask = (image == maskval)
    else:
        alpha_mask = np.ones_like(alpha_image)
    image = (np.clip(image, vmin, vmax) - vmin) / (vmax - vmin)
    image_RGBA = overlay_cmap(image)  # convert to RGBA
    if alpha_mask is not None:
        if alpha_image is None:
            image_RGBA[..., -1][alpha_mask] = 0  # set
        else:
            image_RGBA[..., -1] = image_RGBA[..., -1] * alpha_image
#    if f is None:
#        f = plt.figure()
    plt.imshow(image_RGBA, cmap=overlay_cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(shrink=0.9)
    plt.axis('off')
    plt.axis('image')
    return
#    return f

# fig,(ax1,ax2)=plt.subplots(1,2);
# volshow(T1vols[...,4],ax=ax1,mode='c',show_lines=True);
# volshow(T1vols[...,4],ax=ax2,mode='m',show_lines=True)


def _to_list(y, n):
    """ For list input y, check that len(y) == n.
    For scalar input y, duplicate y into a list of length(n)
    """
    if isinstance(y, (list, tuple, np.ndarray)):
        if len(y) != n:
            raise ValueError("axes list must match length of x")
        return y
    else:
        return [y, ]*n


def _parse_montager_kwargs(kwargs, nd, omit=[], list_only=False):
    """ parse out options specific to the montager """
    montage_args = []
    montage_args += inspect.getargspec(montager)[0]
    if nd > 3:
        # 4d montager has additional options
        montage_args += inspect.getargspec(montager4d)[0]
    for o in omit:
        if o in montage_args:
            montage_args.remove(o)
    if list_only:
        return montage_args
    montage_kwargs = {}
    for k in montage_args:
        if k in kwargs:
            montage_kwargs[k] = kwargs.pop(k)
    montage_kwargs['output_grid_size'] = True
    if 'transpose' not in montage_kwargs:
        montage_kwargs['transpose'] = True
    return montage_kwargs


def _parse_centerplane_kwargs(kwargs, isRGB=False, omit=[], list_only=False):
    """ parse out arguments specific to centerplanes_stack """
    if isRGB:
        stack_func = centerplanes_stack_RGB
    else:
        stack_func = centerplanes_stack
    centerplanes_arg_list = inspect.getargspec(stack_func)[0]
    for o in omit:
        if o in centerplanes_arg_list:
            centerplanes_arg_list.remove(o)
    if list_only:
        return centerplanes_arg_list
    centerplane_kwargs = {}
    for k in centerplanes_arg_list:
        if k in kwargs:
            centerplane_kwargs[k] = kwargs.pop(k)
    return centerplane_kwargs


def _parse_mip_kwargs(kwargs, omit=[], list_only=False):
    """ parse out arguments specific to calc_mips """
    mips_arg_list = inspect.getargspec(calc_mips)[0]
    for o in omit:
        if o in mips_arg_list:
            mips_arg_list.remove(o)
    if list_only:
        return mips_arg_list
    mip_kwargs = {}
    for k in mips_arg_list:
        if k in kwargs:
            mip_kwargs[k] = kwargs.pop(k)
    return mip_kwargs


def volshow(x, mode=None, ax=None, fig=None, subplot=111, cplx_to_abs=True,
            show_lines=False, line_color='w', mask_nan=False, notick=False,
            **kwargs):
    """ volume viewing utility.  Reshapes the volume data based on the desired
    mode and then plots using imshow.

    Parameters
    ----------
    x : array or list of array
        volume to plot.  Also supports a list of volumes.  If a list is passed
        a figure with a number of subplots equal to the legth of the list will
        be created.  If `mode`='imagegrid', an ImageGrid will be used instead
        of subplots.
    mode : {'montage','centerplanes','MIPs','imagegrid','imshow'}, optional
        display mode.  'm','c','p','g', and 'i' are shorthand for 'montage',
        'centerplanes', 'MIPs', 'imagegrid' and 'imshow' respectively.  Default
        is 'montage' for 3D or 4D input, 'imshow' for 2D.
    ax : matplotlib.axes.Axes, optional
        If ax=None and fig=None a new figure is created.  Otherwise, volshow
        will use the provided axis.
    fig : matplotlib.figure.Figure or int or None, optional
        If ax is provided this argument is isnored.  If fig is None, create a
        new figure.  Otherwise, use the specified figure or figure number.
    subplot : int or tuple, optional:
        this will be passed to fig.add_subplot() if `ax`=None.  If `ax` was
        provided `subplot`=ax.get_subplotspec()
    cplx_to_abs : bool, optional
        If true, convert complex data to magnitude before plotting.  Otherwise,
        only the real component is shown.
    show_lines : bool, optional
        Currently only used for mode='montage'.  If true, add lines separating
        the subimages of the montage.
    line_color : str, optional
        the line color to be used by show_lines
    mask_nan : bool, optional
        If true, set all NaN's to zero in the plot
    notick : bool, optional
        If True, suppress all tick marks
    kwargs : additional keyword arguments that will be passed on to the
        the `mode`-dependent plotting routines.  Any kwargs not defined for the
        `mode`-dependent plot routine are passed onto the final matshow() or
        imshow() call

    Other Parameters
    ----------------
    isRGB : bool, optional
        if True and last dimension is size 3 or 4, treat input as a color image
    grid_labels : list of str, optional
        optional kw_arg for subfigure labels when mode='imagegrid'.
    grid_label_kwargs : dict, optional
        additional kw_args specific to `pyvolplot.utils.add_inner_title`.
        These will be applied to all subplots of the imagegrid.


    Returns
    -------
    im : matplotlib.image.AxesImage
        AxesImage object returned by imshow

    """

    if isinstance(x, (list, tuple)):
        # support inputing a list of volumes to each be placed into its own
        # subplot
        if np.all([isinstance(i, np.ndarray) for i in x]):
            if fig is not None:
                warnings.warn("fig argument ignored for list input")
                fig = None

            if isinstance(ax, (list, tuple, np.ndarray)):
                if len(ax) == len(x):
                    axes = np.asarray(ax)
                else:
                    raise ValueError("axes list must match length of x")
            elif ax is not None:
                raise ValueError("single ax not supported for list input. " +
                                 " Using new figure instead.")

            if ax is None:
                # Call volshow() for each individual volume
                srow, scol = _calc_rows(1.0, 1.0, len(x))
                # TODO: support aspect in _calc_rows?
                fig, axes = plt.subplots(srow, scol)
                # disable display of unused axes
                for n in range(len(x), len(fig.axes)):
                    fig.axes[n].set_axis_off()

            # reshape ndarray of axes to 1D and truncate to length of x
            axes = axes.ravel(order='C')[:len(x)]
            im_list = []
            isRGB = kwargs.pop('isRGB', False)

            # check list dimensions or create list from any non-list inputs
            mode = _to_list(mode, len(x))
            cplx_to_abs = _to_list(cplx_to_abs, len(x))
            show_lines = _to_list(show_lines, len(x))
            line_color = _to_list(line_color, len(x))
            mask_nan = _to_list(mask_nan, len(x))
            notick = _to_list(notick, len(x))

            # support lists of other kwargs arguments as well...
            print("kwargs={}".format(kwargs))
            for key in kwargs:
                kwargs[key] = _to_list(kwargs[key], len(x))
            print("kwargs={}".format(kwargs))

            for idx, xi in enumerate(x):
                if isRGB:
                    if isinstance(isRGB, (list, tuple, np.ndarray)):
                        isRGB_subplot = isRGB[idx]
                    else:
                        # attempt to auto-determined RGB flag for each subplot
                        if xi.shape[-1] < 3 or xi.shape[-1] > 4:
                            isRGB_subplot = False
                        else:
                            isRGB_subplot = True
                else:
                    isRGB_subplot = False

                if isRGB_subplot:
                    nd = xi.ndim - 1
                else:
                    nd = xi.ndim

                kwargs_subplot = {}
                for key in kwargs:
                    kwargs_subplot[key] = kwargs[key][idx]
                print("kwargs_subplot={}".format(kwargs_subplot))
                print("kwargs_subplot={}".format(kwargs_subplot))

                # get list of kwargs specific to the current mode
                if mode[idx].lower() in ['m', 'montage']:
                    omit = _parse_montager_kwargs(kwargs_subplot, nd,
                                                  list_only=True)
                elif mode[idx].lower() in ['c', 'centerplanes']:
                    omit = _parse_centerplane_kwargs(kwargs_subplot,
                                                     isRGB_subplot,
                                                     list_only=True)
                elif mode[idx].lower() in ['c', 'centerplanes']:
                    omit = _parse_mip_kwargs(kwargs_subplot, list_only=True)
                elif mode[idx].lower() in ['g', 'imagegrid']:
                    omit = ['transpose', 'grid_labels', 'grid_label_kwargs']
                else:
                    omit = []

                # turn off existing axis if imagegrid will be called
                if mode[idx].lower() in ['g', 'imagegrid']:
                    axes[idx].set_axis_off()

                # strip any kwargs particular to the other modes
                if mode[idx].lower() not in ['m', 'montage']:
                    _parse_montager_kwargs(kwargs_subplot, nd, omit=omit)
                if mode[idx].lower() not in ['c', 'centerplanes']:
                    _parse_centerplane_kwargs(kwargs_subplot,
                                              isRGB_subplot, omit=omit)
                if mode[idx].lower() not in ['p', 'mips']:
                    _parse_mip_kwargs(kwargs_subplot, omit=omit)

                print("kwargs_subplot={}".format(kwargs_subplot))
                im = volshow(xi, mode=mode[idx], ax=axes[idx],
                             cplx_to_abs=cplx_to_abs[idx],
                             show_lines=show_lines[idx],
                             line_color=line_color[idx],
                             mask_nan=mask_nan[idx], notick=notick[idx],
                             isRGB=isRGB_subplot, **kwargs_subplot)
                im_list.append(im)
            return im_list

    x = np.asanyarray(x)
    if np.iscomplexobj(x):
        if cplx_to_abs:
            # print("magnitude of complex data displayed")
            x = np.abs(x)
        else:
            warnings.warn("only real part of complex data displayed")
            x = np.real(x)

    if mask_nan:
        nanmask = np.isnan(x)
        if np.any(nanmask):
            warnings.warn("NaN values found... setting them to 0 for display")
        x[nanmask] = 0

    # if input is RGB or RGBA reduce number of dimensions to tile by 1
    isRGB = kwargs.get('isRGB', False)
    if isRGB:
        if x.dtype == np.uint8:
            pass
        elif x.max() > 1 or x.min() < 0:
            raise ValueError("floating point RGB images must be scaled " +
                             "within [0, 1]")

    if (x.ndim >= 3) and isRGB:
        nd = x.ndim - 1
    else:
        nd = x.ndim

    if nd < 2:
        warnings.warn(
            "input only has {} dimensions. converting to 2D".format(nd))
        x = np.atleast_2d(x)

    if mode is None:
        if nd >= 3:
            mode = 'montage'
        else:
            mode = 'imshow'

    # generate a new figure if an existing axis was not passed in
    if ax is None:
        if fig is None:
            # generate new figure
            fig = plt.figure()
        elif isinstance(fig, matplotlib.figure.Figure):
            pass
        elif isinstance(fig, int):
            fig = plt.figure(fig)
        else:
            raise ValueError("bad input type: {} for fig".format(type(fig)))
        if mode.lower() not in ['g', 'imagegrid']:
            ax = fig.add_subplot(subplot)
    else:
        if fig is not None:
            # raise ValueError("cannot specify both fig and ax")
            warnings.warn("using fig from specified axis instead of provided",
                          "fig input")
        fig = ax.get_figure()
        subplot = ax.get_subplotspec()

    if not is_string_like(mode):
        raise ValueError("mode must be string-like")

    if mode.lower() in ['m', 'montage']:
        if nd > 4:
            warnings.warn(
                "montager only tiles up to 4D.  all excess dimensions" +
                "collapsed into 4th (Fortran order)")
            x = x.reshape(x.shape[:3] + (-1,), order='F')
        # separate out options to the montager
        # assume remainder of kwargs go to imshow
        montage_kwargs = _parse_montager_kwargs(kwargs, nd=nd)
        montage_kwargs['xi'] = x

        tmp = montage_kwargs.get('transpose', True)
        if (tmp and nd >= 3) or (not tmp and nd < 3):
            xticks = [x.shape[0], ]
            yticks = [x.shape[1], ]
        else:
            yticks = [x.shape[0], ]
            xticks = [x.shape[1], ]

        if nd >= 4:
            x, nrows, ncols = montager4d(**montage_kwargs)
        else:
            x, nrows, ncols = montager(**montage_kwargs)

    elif mode.lower() in ['c', 'centerplanes']:
        if nd != 3:
            raise ValueError("centerplanes mode only supports 3D volumes")
        if isRGB:
            stack_func = centerplanes_stack_RGB
        else:
            stack_func = centerplanes_stack
        centerplane_kwargs = _parse_centerplane_kwargs(kwargs, isRGB)
        centerplane_kwargs['x'] = x
        x = stack_func(**centerplane_kwargs)

    elif mode.lower() in ['p', 'mips']:
        if isRGB:
            raise ValueError("RGB not support for MIP mode")
        mip_kwargs = _parse_mip_kwargs(kwargs)
        mip_kwargs['x'] = x
        x = calc_mips(**mip_kwargs)

    elif mode.lower() in ['i', 'imshow']:
        if nd > 2:
            raise ValueError("imshow mode only works for 2D input")
    elif mode.lower() in ['g', 'imagegrid']:
        if x.ndim > 3:
            if isRGB:
                x = np.reshape(x, (x.shape[0], x.shape[1], -1, x.shape[-1]))
            else:
                x = np.reshape(x, (x.shape[0], x.shape[1], -1))
        elif x.ndim == 2:
            x = x[:, :, np.newaxis]
        elif x.ndim == 1:
            x = x[:, np.newaxis, np.newaxis]

        if x.shape[-1] > 25:
            warnings.warn("imagegrid mode likely to be slow when number of",
                          "subplots is large, consider using mode='montage'")

        grid_labels = kwargs.pop('grid_labels', [])
        if grid_labels:
            if not isinstance(grid_labels, (list, tuple)):
                raise ValueError("grid_labels must be a list or tuple")
            if not np.all([is_string_like(l) for l in grid_labels]):
                raise ValueError("grid_labels must contain onlt strings")
            default_grid_label_kwargs = dict(loc=9, size='large')
            grid_label_kwargs = kwargs.pop('grid_label_kwargs', {})
            for key, val in default_grid_label_kwargs.items():
                if key not in grid_label_kwargs:
                    grid_label_kwargs[key] = val
            if 'title' in grid_label_kwargs:
                warnings.warn("'title' field of grid_label_kwargs unsupported",
                              "use grid_labels to pass in the labels")
                grid_label_kwargs.pop('title', None)

        row, col = _calc_rows(*x.shape[:3])
        grid = ImageGrid(fig, subplot, nrows_ncols=(row, col), axes_pad=.05)

        if 'transpose' not in kwargs:
            kwargs['transpose'] = True

        for iz in range(x.shape[2]):
            if isRGB:
                if kwargs['transpose']:
                    grid[iz].imshow(x[:, :, iz, :])
                else:
                    grid[iz].imshow(x[:, :, iz, :].transpose((1, 0, 2)))
            else:
                if kwargs['transpose']:
                    grid[iz].imshow(x[:, :, iz].T)
                else:
                    grid[iz].imshow(x[:, :, iz])
            if iz == 0:
                if kwargs['transpose']:
                    grid[iz].set_xticks([x.shape[0]])
                else:
                    grid[iz].set_xticks([x.shape[1]])
                grid[iz].xaxis.tick_top()
                if kwargs['transpose']:
                    grid[iz].set_yticks([x.shape[1]])
                else:
                    grid[iz].set_yticks([x.shape[0]])
            else:
                grid[iz].axis('off')

        if grid_labels:
            for iz in range(x.shape[2]):
                if iz < len(grid_labels):
                    t = add_inner_title(ax=grid[iz], title=grid_labels[iz],
                                        **grid_label_kwargs)
                    t.patch.set_alpha(0.5)

        plt.show()
        return fig, grid
    else:
        raise ValueError("unsupported mode")

    # change a few of the imshow defaults
    if 'cmap' not in kwargs:
        kwargs['cmap'] = plt.cm.gray
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'

    # don't pass on volshow-specific keywords to imshow() or matshow()
    kwargs.pop('isRGB', None)
    kwargs.pop('grid_labels', None)
    kwargs.pop('grid_label_kwargs', None)

    if isRGB:
        # matshow doesn't support color images
        im = ax.imshow(x, **kwargs)
        # adjust axes & aspect to be similar to matshow() case
        aspect = matplotlib.rcParams['image.aspect']
        ax.set_aspect(aspect)
        ax.xaxis.tick_top()
    else:
        im = ax.matshow(x, **kwargs)

    # if requested, draw lines dividing the cells of the montage
    if show_lines and (mode.lower() in ['m', 'montage']):
        # TODO: add lines for MIP or centerplane cases
        add_lines(x,
                  color=line_color,
                  ncells_horizontal=ncols,
                  ncells_vertical=nrows)

    if notick:  # remove axis tick marks
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if mode in ['m', 'montage']:
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            if show_lines:  # match tickmark color to line_color
                ax.tick_params(axis='both', color=line_color)
            else:
                ax.tick_params(axis='both', color='w')
        else:
            ax.tick_params(axis='both', color='w')

    return im


def test_volshow_list():
    from pyvolplot import volshow
    import skimage.data
    cat = skimage.data.chelsea()
    coins = skimage.data.coins()
    volshow([cat, coins, cat],
            isRGB=[True, False, 0], mode=['i', 'm', 'm'], transpose=False)

    volshow([cat, coins, cat, cat],
            isRGB=[1, 0, 0, 0], mode=['i', 'm', 'm', 'g'], transpose=False)
