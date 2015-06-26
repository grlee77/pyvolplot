import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter3(x, y=None, z=None, ax=None, view_init=[0, 0], limits=None,
             pos_x_only=False, aspect='equal', **kwargs):
    """ convenience interface to scatter3D that allows x to be [3 x N] or [N x 3].

    Parameters
    ----------
    x : np.ndarray
        x coordinates.  If y, z not supplied can be an [N x 3] or [3 x N] array
        of x,y,z coordinates
    y : np.ndarray, optional
        y coordinates
    z : np.ndarray, optional
        z coordinates
    ax : axis, optional
        axis to use for the plot
    view_init : list[2], optional
        initial view angle, optional
    limits : tuple of tuples
        view limits:  ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    pos_x_only : bool, optional
        if True, omit any points with a negative x coordinate
    aspect : str, optional
        figure aspect
    kwargs : dict
        other kwargs to pass on to scatter3D

    Returns
    -------
    ax : axis
        axis containing the scatter plot

    Note
    ----
    If x, y, and z are all 2D arrays, the plot will use separate colors for
    each column.

    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if y is None or z is None:
        if y is not None or z is not None:
            raise ValueError("must supply either both y and z or neither")
        if x.ndim == 2:
            if x.shape[1] == 3:
                if pos_x_only:
                    slc = np.where(x[:, 0] > 0)[0]
                else:
                    slc = slice(None)
                ax.scatter3D(x[slc, 0], x[slc, 1], x[slc, 2])
        else:
            raise ValueError("x must be 2D with (x.shape[1] == 3) if y and z "
                             "are not provided")
    else:
        if y.shape != x.shape or z.shape != x.shape:
            raise ValueError("x, y, z must have matching shape")
        if x.ndim == 1:
            ax.scatter3D(x, y, z)
        else:
            colors = kwargs.pop('colors', 'bgrcmyk')  # TODO use a random color instead
            ncolors = len(colors)
            nplots = x.shape[1]
            for n in range(nplots):
                color = colors[np.mod(n, ncolors)]
                if pos_x_only:
                    slc = np.where(x[:, n] > 0)[0]
                else:
                    slc = slice(None)
                ax.scatter3D(x[slc, n], y[slc, n], z[slc, n], c=color)

    if view_init is not None:
        ax.view_init(view_init[0], view_init[1])
    if aspect is not None:
        ax.set_aspect('equal')
    if limits is not None:
        ax.set_xlim3d(limits[0][0], limits[0][1])
        ax.set_ylim3d(limits[1][0], limits[1][1])
        ax.set_zlim3d(limits[2][0], limits[2][1])
    return ax
