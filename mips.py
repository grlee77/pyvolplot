from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np


def calc_mips(x, stack_direction='h', transpose=True):
    x = np.asanyarray(x)
    if x.ndim != 3:
        raise ValueError("only 3D data supported")
    if transpose:
        x = np.transpose(x, [2, 1, 0])
    if stack_direction == 'h':

        axes_to_pad = np.arange(0, x.ndim - 1, dtype=np.int)
        ytarget_size = np.max(x.shape[0:-1])
        ypad_widths = []
        for a in axes_to_pad:
            padw = max(ytarget_size - x.shape[a], 0)
            if padw % 2 == 0:
                ypad_widths.append((padw / 2, padw / 2))
            else:
                ypad_widths.append((padw // 2, padw - padw // 2))
        ypad_widths.append((0, 0))

        xout = np.pad(np.max(np.abs(x), axis=0),
                      pad_width=(ypad_widths[1], ypad_widths[2]),
                      mode='constant', constant_values=0)
        x2 = np.pad(np.max(np.abs(x), axis=1),
                    pad_width=(ypad_widths[0], ypad_widths[2]),
                    mode='constant', constant_values=0)
        x3 = np.pad(np.max(np.abs(x), axis=2),
                    pad_width=(ypad_widths[0], ypad_widths[1]),
                    mode='constant', constant_values=0)
        xout = np.concatenate((xout, x2), axis=1)
        xout = np.concatenate((xout, x3), axis=1)
    elif stack_direction == 'v':
        axes_to_pad = np.arange(0, x.ndim)
        xtarget_size = np.max(x.shape[1::])
        xpad_widths = []
        for a in axes_to_pad:
            padw = max(xtarget_size - x.shape[a], 0)
            if padw % 2 == 0:
                xpad_widths.append((padw / 2, padw / 2))
            else:
                xpad_widths.append((padw // 2, padw - padw // 2))
        # xpad_widths.append((0,0))
        xout = np.pad(np.max(np.abs(x), axis=0),
                      pad_width=(xpad_widths[1], xpad_widths[2]),
                      mode='constant', constant_values=0)
        x2 = np.pad(np.max(np.abs(x), axis=1),
                    pad_width=(xpad_widths[0], xpad_widths[2]),
                    mode='constant',  constant_values=0)
        x3 = np.pad(np.max(np.abs(x), axis=2),
                    pad_width=(xpad_widths[0], xpad_widths[1]),
                    mode='constant', constant_values=0)
        xout = np.concatenate((xout, x2), axis=0)
        xout = np.concatenate((xout, x3), axis=0)

    return xout


def plot_mips(x, stack_direction='h', open_newfig=False, **kwargs):

    # TODO: shrink borders, etc...

    # MIPs don't necessarily have matching size, so manually pad the axes
    # rather than just using montager

    xout = calc_mips(x, stack_direction=stack_direction)

    if open_newfig:
        plt.figure()

    plt.imshow(xout, **kwargs), plt.axis('off'), plt.axis('image')
