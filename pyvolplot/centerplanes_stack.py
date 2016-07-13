# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np


def centerplanes_stack_RGB(x, stack_direction='h', transpose=True,
                           slice_centers=None, npad=0):
    if x.ndim != 4:
        raise ValueError("requires 4D input")
    if x.shape[3] < 3 or x.shape[3] > 4:
        raise ValueError("4th dimension must be size 3 or 4 (RGB or RGBA)")

    R = centerplanes_stack(x[..., 0], stack_direction=stack_direction,
                           npad=npad, transpose=transpose)

    xout = np.zeros(R.shape + (x.shape[3],), x.dtype)
    xout[:, :, 0] = R
    for n in range(x.shape[3]):
        xout[:, :, n] = centerplanes_stack(x[:, :, :, n],
                                           stack_direction=stack_direction,
                                           npad=npad,
                                           transpose=transpose)
    return xout


def centerplanes_stack(x, stack_direction='h', transpose=True,
                       slice_centers=None, npad=0):

    if x.ndim != 3:
        raise ValueError("requires 3D input")
    if (npad < 0) or (not np.isscalar(npad)) or (np.remainder(npad, 1) > 0):
        raise ValueError("npad must be a positive integer")
    # TODO: shrink borders, etc...
    if transpose:
        x = np.transpose(x, [2, 1, 0])
    # don't necessarily have matching size, so manually pad the axes
    # rather than just using montager
    if slice_centers is None:  # central slices by default
        slice_centers = np.asarray(x.shape) // 2
    else:
        if len(slice_centers) != 3:
            raise ValueError("slice_centers must be length 3")

    if stack_direction == 'h':

        axes_to_pad = np.arange(0, x.ndim - 1, dtype=np.int)
        ytarget_size = np.max(x.shape[0:-1])
        ypad_widths = []
        for a in axes_to_pad:
            padw = max(ytarget_size - x.shape[a], 0) + 2 * npad
            if padw % 2 == 0:
                ypad_widths.append((padw // 2, padw // 2))
            else:
                ypad_widths.append((padw // 2, padw - padw // 2))
        ypad_widths.append((npad, npad))

        xout = np.pad(np.abs(x[slice_centers[0], :, :]),
                      pad_width=(ypad_widths[1], ypad_widths[2]),
                      mode='constant', constant_values=0)
        x2 = np.pad(np.abs(x[:, slice_centers[1], :]),
                    pad_width=(ypad_widths[0], ypad_widths[2]),
                    mode='constant', constant_values=0)
        x3 = np.pad(np.abs(x[:, :, slice_centers[2]]),
                    pad_width=(ypad_widths[0], ypad_widths[1]),
                    mode='constant', constant_values=0)
        xout = np.concatenate((xout, x2), axis=1)
        xout = np.concatenate((xout, x3), axis=1)
    elif stack_direction == 'v':
        axes_to_pad = np.arange(0, x.ndim)
        xtarget_size = np.max(x.shape[1::])
        xpad_widths = []
        for a in axes_to_pad:
            padw = max(xtarget_size - x.shape[a], 0) + 2 * npad
            if padw % 2 == 0:
                xpad_widths.append((padw // 2, padw // 2))
            else:
                xpad_widths.append((padw // 2, padw - padw // 2))
        xout = np.pad(np.abs(x[slice_centers[0], :, :]),
                      pad_width=(xpad_widths[1], xpad_widths[2]),
                      mode='constant', constant_values=0)
        x2 = np.pad(np.abs(x[:, slice_centers[1], :]),
                    pad_width=(xpad_widths[0], xpad_widths[2]),
                    mode='constant', constant_values=0)
        x3 = np.pad(np.abs(x[:, :, slice_centers[2]]),
                    pad_width=(xpad_widths[0], xpad_widths[1]),
                    mode='constant', constant_values=0)
        xout = np.concatenate((xout, x2), axis=0)
        xout = np.concatenate((xout, x3), axis=0)

    return xout
