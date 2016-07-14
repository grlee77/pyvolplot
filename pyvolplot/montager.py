# -*- coding: utf-8 -*-
"""
**Create Image Montage**
"""

# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np


def add_lines(e, color='w', ncells_horizontal=1, ncells_vertical=1):
    """ takes output from montager and the number of rows/cols and plots lines
    separating the cells """

    e = np.asanyarray(e)
    if e.ndim != 2:
        raise ValueError("Requires 2D array")
    if ncells_horizontal > 1:
        s = e.shape[1]
        line_locs = np.round(np.arange(1, ncells_horizontal) *
                             s / ncells_horizontal)
        for l in line_locs:
            plt.plot([l, l], [0, e.shape[0]], color)
            plt.axis('image')
    if ncells_horizontal > 1:
        s = e.shape[0]
        line_locs = np.round(
            np.arange(1, ncells_vertical) * s / ncells_vertical)
        for l in line_locs:
            plt.plot([0, e.shape[1]], [l, l], color)
            plt.axis('image')


def _calc_rows(nx, ny, nz, row=None, col=None, aspect=1.4):
    """ utility to calculate the desired number of rows & columns for the
    image montage """
    if not col:
        if not row:
            col = np.round(np.sqrt(nz * ny / nx * aspect))
        else:
            col = np.ceil(nz / row)
        col = col
    if not row:
        row = np.ceil(nz / col)
    if row * col < nz:
        raise ValueError(
            "Specified {} rows and {} cols".format(row, col) +
            " which is too few to hold {} subimages".format(nz))
    return int(row), int(col)


def montager4d(
        xi, axis=-1, row2=None, col2=None, aspect2=None, **montager_args):
    """ nested montager for 4D data.
        montage each 3D subvolume, then montage the montages
    """
    # TODO: support ndim=5 for RGBA input
    # TODO?: allow input < size 4, just pad size to ones on missing axes?
    xi = np.asanyarray(xi)

    isRGB = montager_args.get('isRGB', False)
    if isRGB:
        if xi.shape[-1] < 3 or xi.shape[-1] > 4:
            raise ValueError(
                "if isRGB=True, the last dimension must be size 3 or 4")
        if xi.shape[-1] == 4:
            has_alpha = True
        else:
            has_alpha = False

        xiR = xi[..., 0]
        xiG = xi[..., 1]
        xiB = xi[..., 2]

        montage2_args = montager_args.copy()
        montage2_args['output_grid_size'] = True
        xoR, row2, col2 = montager4d(xiR, col2=col2, row2=row2, aspect2=aspect2,
                                     axis=axis, **montage2_args)
        xoR = xoR[..., np.newaxis]
        montage2_args['output_grid_size'] = False
        xoG = montager4d(xiG, col2=col2, row2=row2, aspect2=aspect2,
                         axis=axis, **montager_args)
        xoG = xoG[..., np.newaxis]
        xoB = montager(xiB, col2=col2, row2=row2, aspect2=aspect2,
                       axis=axis, **montager_args)
        xoB = xoB[..., np.newaxis]
        if has_alpha:
            xiA = xi[..., 3]
            xoA = montager(xiA, col2=col2, row2=row2, aspect2=aspect2,
                           axis=axis, **montage2_args)
            xoA = xoA[..., np.newaxis]
            xo = np.concatenate((xoR, xoG, xoB, xoA), axis=-1)
        else:
            xo = np.concatenate((xoR, xoG, xoB), axis=-1)
        if montager_args.get('output_grid_size', False):
            return (xo, row2, col2)
        else:
            return xo

    if xi.ndim != 4:
        raise ValueError("montager4d requires 4d input")
    #if montager_args.get('isRGB', False):
    #    raise ValueError("isRGB=True not currently supported")

    nvols = xi.shape[axis]
    slices = [slice(None), ] * 4
    slices[axis] = 0
    if montager_args.get('output_grid_size', False):
        m0 = montager(xi[slices], **montager_args)[0]
    else:
        m0 = montager(xi[slices], **montager_args)

    m_out = np.zeros(m0.shape + (nvols,), dtype=xi.dtype)
    m_out[:, :, 0] = m0
    for n in range(1, nvols):
        slices[axis] = n
        if montager_args.get('output_grid_size', False):
            m_out[:, :, n] = montager(xi[slices], **montager_args)[0]
        else:
            m0 = montager(xi[slices], **montager_args)
    montage2_args = montager_args.copy()
    # flip and transpose operations skipped on second time through montager
    montage2_args['transpose'] = False
    montage2_args['flipx'] = False
    montage2_args['flipy'] = False
    montage2_args['flipz'] = False
    #montage2_args['isRGB'] = False
    montage2_args['col'] = col2
    montage2_args['row'] = row2
    if aspect2 is not None:
        montage2_args['aspect'] = aspect2
    if montager_args.get('output_grid_size', False):
        return montager(m_out, **montage2_args)
    else:
        return (montager(m_out, **montage2_args)[0], row2, col2)


def montager(xi, col=None, row=None, aspect=1.4, transpose=False, isRGB=False,
             flipx=False, flipy=False, flipz=False, output_grid_size=False):
    """ tile a 3D or 4D image into a single 2D montage

    Parameters
    ----------
    xi : ndarray
        image data to montage
    col : int, optional
        number of columns in the montage
    row : int, optional
        number of rows in the montage
    aspect : float, optional
        desired aspect ratio of the montage
    transpose : bool, optional
        transpose each image slice in the montage? (transposes first two
        dimensions of the input)
    isRGB : bool, optional
        set True if the input is RGB
    flipx : bool, optional
        reverse x-axis indices?
    flipy : bool, optional
        reverse y-axis indices?
    flipz : bool, optional
        reverse z-axis indices?
    output_grid_size : bool, optional
        if true, the number of rows and columns will also be returned

    Returns
    -------
    xo : ndarray
        2D ndarray containing the montage

    Notes
    -----
    Any axis flips are applied prior to transposition
    added RGB support, aspect ratio, transpose flag and axis flip flags
    adapted from:  montager.m  (Jeff Fessler's IRT toolbox)

    """
    # TODO?: also allow RGBA axis to be the first rather than last
    # TODO: add option to add a border between the cells
    # TODO: allow >4D by stacking all remaining dimensions along the 4th

    if isRGB:  # call montager for R,G,B channels separately
        if xi.shape[-1] < 3 or xi.shape[-1] > 4:
            raise Exception("if isRGB=True, the last dimension must be size 3 or 4")
        if xi.shape[-1] == 4:
            has_alpha = True
        else:
            has_alpha = False

        xiR = xi[..., 0]
        xiG = xi[..., 1]
        xiB = xi[..., 2]
        xoR, row, col = montager(xiR, col=col, row=row, aspect=aspect,
                                 transpose=transpose, isRGB=False,
                                 flipx=flipx, flipy=flipy, flipz=flipz,
                                 output_grid_size=True)
        xoR = xoR[:, :, None]
        xoG = montager(xiG, col=col, row=row, aspect=aspect,
                       transpose=transpose, isRGB=False, flipx=flipx,
                       flipy=flipy, flipz=flipz,
                       output_grid_size=False)
        xoG = xoG[:, :, None]
        xoB = montager(xiB, col=col, row=row, aspect=aspect,
                       transpose=transpose, isRGB=False, flipx=flipx,
                       flipy=flipy, flipz=flipz, output_grid_size=False)
        xoB = xoB[:, :, None]
        if has_alpha:
            xiA = xi[..., 3]
            xoA = montager(xiA, col=col, row=row, aspect=aspect,
                           transpose=transpose, isRGB=False, flipx=flipx,
                           flipy=flipy, flipz=flipz, output_grid_size=False)
            xoA = xoA[:, :, None]
            xo = np.concatenate((xoR, xoG, xoB, xoA), axis=2)
        else:
            xo = np.concatenate((xoR, xoG, xoB), axis=2)
        if output_grid_size:
            return (xo, row, col)
        else:
            return xo

    if xi.ndim > 4:
        print('ERROR in %s: >4D not done' % __name__)
    if xi.ndim == 4:
        if flipx:
            xi = xi[::-1, :, :, :]
        if flipy:
            xi = xi[:, ::-1, :, :]
        if flipz:
            xi = xi[:, :, ::-1, :]
        if not transpose:
            xi = np.transpose(xi, axes=(1, 0, 2, 3))
        (nx, ny, n3, n4) = xi.shape
        nz = n3 * n4
        xi = np.reshape(xi, (nx, ny, nz), order='F')
    elif xi.ndim == 3:
        if flipx:
            xi = xi[::-1, :, :]
        if flipy:
            xi = xi[:, ::-1, :]
        if flipz:
            xi = xi[:, :, ::-1]
        if not transpose:
            xi = np.transpose(xi, axes=(1, 0, 2))
        (nx, ny, nz) = xi.shape
    else:  # for 1D or 2D case, just return the input, unchanged
        if flipx:
            xi = xi[::-1, :]
        if flipy:
            xi = xi[:, ::-1]
        if not transpose:
            xi = xi.T
        if output_grid_size:
            return xi, 1, 1
        else:
            return xi

    if xi.ndim == 4:
        col = n3

    row, col = _calc_rows(nx, ny, nz, row=row, col=col, aspect=aspect)

    xo = np.zeros((ny * row, nx * col))

    for iz in range(nz):
        iy = int(np.floor(iz / col))
        ix = iz - iy * col
        xo[iy * ny:(iy + 1) * ny, ix * nx:(ix + 1) * nx] = xi[:, :, iz].T

    if output_grid_size:
        return (xo, row, col)
    else:
        return xo


def montager_test():
    t = (20, 30, 5)
    t = np.reshape(np.arange(0, np.prod(t)), t, order='F')
    # order='F' to match matlab behavior
    plt.figure()
    plt.subplot(121)
    plt.imshow(montager(t), cmap=plt.cm.gray, interpolation='nearest')
    plt.subplot(122)
    plt.imshow(montager(t, row=1), cmap=plt.cm.gray, interpolation='nearest')
