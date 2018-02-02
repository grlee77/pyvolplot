# -*- coding: utf-8 -*-
from __future__ import absolute_import

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyvolplot.montager import montager, montager4d


def cycle_frames(img_vols, time_axis=-1, anim_kw=dict(interval=50, blit=True),
                 imshow_kw={}):
    """Make a video loop of image frames cycled along `time_axis`.

    Parameters
    ----------
    img_vols : array
        Should be 3D, 4D or 5D.  The 4D and 5D cases will have the data tiled
        into a 2D view in each frame via `montager`.
    time_axis : int, optional
        Can be used to specify which array axis to treat as the time
        dimension.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        Matplotlib FuncAnimation object.

    """
    ndim = img_vols.ndim
    if ndim < 3 or ndim > 5:
        raise ValueError("input data must be 3D, 4D or 5D")
    if ndim < 5:
        montage_func = montager
    elif ndim == 5:
        montage_func = montager4d

    slices = [slice(None), ] * img_vols.ndim

    fig = plt.figure()
    fig.patch.set_visible = False

    frame = 0
    if 'cmap' not in imshow_kw:
        imshow_kw['cmap'] = plt.get_cmap('gray')
    slices[time_axis] = frame
    nframes = img_vols.shape[-1]
    im = plt.imshow(montage_func(img_vols[slices]),
                    **imshow_kw)
    plt.axis('off')
    im.axes.set_visible = False

    def updatefig(frame, *args):
        frame = frame % nframes
        slices[time_axis] = frame
        im.set_array(montage_func(img_vols[slices]))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, **anim_kw)
    plt.show()
    return ani


def cycle_frames_overlay(bg_img, img_vols, time_axis=-1, anim_kw=dict(interval=50, blit=True),
                         imshow_kw={}, alpha_image=None):
    """Make a video loop of image frames cycled along `time_axis`.

    Parameters
    ----------
    img_vols : array
        Should be 3D, 4D or 5D.  The 4D and 5D cases will have the data tiled
        into a 2D view in each frame via `montager`.
    time_axis : int, optional
        Can be used to specify which array axis to treat as the time
        dimension.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        Matplotlib FuncAnimation object.

    """
    ndim = img_vols.ndim
    if ndim < 3 or ndim > 5:
        raise ValueError("input data must be 3D, 4D or 5D")
    if ndim < 5:
        montage_func = montager
    elif ndim == 5:
        montage_func = montager4d

    slices = [slice(None), ] * img_vols.ndim

    fig = plt.figure()
    fig.patch.set_visible = False

    frame = 0
    if 'cmap' not in imshow_kw:
        imshow_kw['cmap'] = plt.get_cmap('gray')
    slices[time_axis] = frame
    nframes = img_vols.shape[-1]
    im = plt.imshow(montage_func(img_vols[slices]),
                    **imshow_kw)
    plt.axis('off')
    im.axes.set_visible = False

    def updatefig(frame, *args):
        frame = frame % nframes
        slices[time_axis] = frame
        im.set_array(montage_func(img_vols[slices]))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, **anim_kw)
    plt.show()
    return ani
