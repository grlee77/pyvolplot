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
from matplotlib import  is_string_like


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


def volshow(x, mode=None, ax=None, cplx_to_abs=True, show_lines=False,
            line_color='w', mask_nan=False, notick=False, **kwargs):
    """ volume viewing utility.  Reshapes the volume data based on the desired
    mode and then plots using imshow.

    Inputs
    ------
    x : np.ndarray
        volume to plot
    mode : {'m','montage','c','centerplanes','p','MIPs','i','imshow'}, optional
        display mode.  'm','c','p' are shorthand for 'montage','centerplanes',
        'MIPs'.  Default = 'montage' for 3D or 4D input, 'imshow' for 2D.
    ax : matplotlib.axes.Axes, optional
        If ax=None, a new figure is created.  Other imshow is called on the
        provided axis
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
        If True,

    Returns
    -------
    im : matplotlib.image.AxesImage
        AxesImage object returned by imshow

    """

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
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    elif mode.lower() in ['g', 'imagegrid']:
        raise ValueError("User supplied axis not supported in ImageGrid mode")


    if not is_string_like(mode):
        raise ValueError("mode must be string-like")

    if mode.lower() in ['m', 'montage']:
        if nd > 4:
            warnings.warn(
                "montager only tiles up to 4D.  all excess dimensions" +
                "collapsed into 4th (Fortran order)")
            x = x.reshape(x.shape[:3] + (-1,), order='F')

        # separate out options to the montager
        montage_args = []
        montage_args += inspect.getargspec(montager)[0]
        if nd > 3:
            # 4d montager has additional options
            montage_args += inspect.getargspec(montager4d)[0]

        montage_kwargs = {}
        for k in montage_args:
            if k in kwargs:
                montage_kwargs[k] = kwargs.pop(k)

        montage_kwargs['output_grid_size'] = True
        montage_kwargs['xi'] = x

        if 'transpose' not in montage_kwargs:
            montage_kwargs['transpose'] = True

        # assume remainder of kwargs go to imshow
        if montage_kwargs.get('transpose', True):
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

        centerplanes_arg_list = inspect.getargspec(stack_func)[0]
        centerplane_kwargs = {}
        for k in centerplanes_arg_list:
            if k in kwargs:
                centerplane_kwargs[k] = kwargs.pop(k)
        centerplane_kwargs['x'] = x

        x = stack_func(**centerplane_kwargs)

    elif mode.lower() in ['p', 'mips']:
        if isRGB:
            raise ValueError("RGB not support for MIP mode")

        mips_arg_list = inspect.getargspec(calc_mips)[0]
        mip_kwargs = {}
        for k in mips_arg_list:
            if k in kwargs:
                mip_kwargs[k] = kwargs.pop(k)
        mip_kwargs['x'] = x
        x = calc_mips(**mip_kwargs)

    elif mode.lower() in ['i', 'imshow']:
        if nd > 2:
            raise ValueError("imshow mode only works for 2D input")
    elif mode.lower() in ['g', 'imagegrid']:
        warnings.warn("imagegrid case experimental. not recommended")
        if x.ndim > 3:
            if isRGB:
                x = np.reshape(x, (x.shape[0], x.shape[1], -1, x.shape[-1]))
            else:
                x = np.reshape(x, (x.shape[0], x.shape[1], -1))
        f = plt.figure()
        row, col = _calc_rows(*x.shape[:3])
        grid = ImageGrid(f, 111, nrows_ncols=(row, col), axes_pad=.05)
        for iz in range(x.shape[2]):
            if isRGB:
                grid[iz].imshow(x[:, :, iz, :].transpose((1, 0, 2)))
            else:
                grid[iz].imshow(x[:, :, iz].T)
            if iz == 0:
                grid[iz].set_xticks([x.shape[0]])
                grid[iz].xaxis.tick_top()
                grid[iz].set_yticks([x.shape[1]])
            else:
                grid[iz].axis('off')
        plt.show()
        return f, grid
    else:
        raise ValueError("unsupported mode")

    # change a few of the imshow defaults
    if 'cmap' not in kwargs:
        kwargs['cmap'] = plt.cm.gray
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'
    # don't pass on isRGB keyword
    kwargs.pop('isRGB', None)

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
