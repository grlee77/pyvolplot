from __future__ import division, print_function, absolute_import
import numpy as np
from matplotlib import pyplot as plt


def masked_overlay(overlay_image, ax=None, overlay_cmap=plt.cm.hot,
                   add_colorbar='False', colorbar_kwargs={'shrink': 0.9},
                   vmin=None, vmax=None, alpha=1.0, alpha_image=None,
                   maskval=0):
    """ overlay another volume via alpha transparency onto the existing volume
    plotted on axis ax.

    Parameters
    ----------
    overlay_image : np.ndarray
        volume to use for the overlay
    ax : matplotlib.axes.Axes, optional
        axis to add the overlay to.  plt.gca() if unspecified
    overlay_cmap : matplotlib.colors.Colormap
        colormap for the overlay
    add_colorbar : bool, optional
        determine of a colorbar should be added to the axis
    colorbar_kwargs : dict, optional
        additional arguments to pass on to the colorbar
    vmin : float, optional
        minimum value for imshow.  alpha = 0 anywhere `overlay_image` < `vmin`
    vmax : float, optional
        maximum value for imshow
    alpha_image : np.ndarray, optional
        if provided, use this as the transparency channel for the overlay
    alpha : float, optional
        transparency of the overlay is equal to alpha, unless `alpha_image` is
        provided instead
    maksval : float, optional
        anywhere `overlay_image` == `maskval`, alpha = 0
    """

    if vmin is None:
        vmin = overlay_image.min()
    if vmax is None:
        vmax = overlay_image.max()
    if alpha_image is None:
        if maskval is not None:
            alpha_mask = (overlay_image == maskval)
    else:
        if alpha_image.max() > 1 or alpha_image.min() < 0:
            raise ValueError("alpha_image must lie in range [0, 1]")
        alpha_mask = np.ones_like(alpha_image)
        # alpha_mask[overlay_image < vmin] = 0
    alpha_mask = alpha_mask | (overlay_image < vmin)
    image = (np.clip(overlay_image, vmin, vmax) - vmin) / (vmax - vmin)
    image_RGBA = overlay_cmap(image)  # convert to RGBA
    if alpha_mask is not None:
        if alpha_image is None:
            image_RGBA[..., -1][alpha_mask] = 0  # set
            image_RGBA[..., -1] *= alpha
        else:
            image_RGBA[..., -1] = image_RGBA[..., -1] * alpha_image

    im = ax.imshow(image_RGBA, cmap=overlay_cmap, vmin=vmin, vmax=vmax)
    if add_colorbar:
        plt.colorbar(im, ax=ax, **colorbar_kwargs)
    ax.axis('off')
    ax.axis('image')
    return
