# -*- coding: utf-8 -*-
from __future__ import absolute_import

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .montager import montager, montager4d


def cycle_frames(img_vols, time_axis=-1):
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

    frame = 0
    slices[time_axis] = frame
    nframes = img_vols.shape[-1]
    im = plt.imshow(montage_func(img_vols[slices]),
                    cmap=plt.get_cmap('gray'))

    def updatefig(*args):
        global frame
        frame += 1
        frame = frame % nframes
        slices[time_axis] = frame
        im.set_array(montage_func(img_vols[slices]))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    plt.show()
    return ani
