# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.ndimage import affine_transform

"""
slight modification of dipy.align.aniso2iso.resample
unlike the dipy version, this one works with complex data and does not require
an affine
"""


def resample_image(data, zooms, new_zooms, affine=None, order=1,
                   mode='constant', cval=0, time_axis=-1):
    ''' Resample data from anisotropic to isotropic voxel size

    Parameters
    ----------
    data : array, shape (I,J,K) or (I,J,K,N)
        3d volume or 4d volume with datasets
    zooms : tuple, shape (3,)
        voxel size for (i,j,k) dimensions
    new_zooms : tuple, shape (3,)
        new voxel size for (i,j,k) after resampling
    affine : array, shape (4,4), optional
        mapping from voxel coordinates to world coordinates
    order : int, from 0 to 5, optional
        order of interpolation for resampling/reslicing,
        0 nearest interpolation, 1 trilinear etc..
        if you don't want any smoothing 0 is the option you need.
    mode : string ('constant', 'nearest', 'reflect' or 'wrap'), optional
        Points outside the boundaries of the input are filled according
        to the given mode.
    cval : float, optional
        Value used for points outside the boundaries of the input if
        mode='constant'.
    time_axis : int, optional
        if input is 4D, this determines the axis to loop over


    Returns
    -------
    data2 : array, shape (I,J,K) or (I,J,K,N)
        datasets resampled into isotropic voxel size
    affine2 : array, shape (4,4), optional
        new affine for the resampled image

    Notes
    -----
    It is also possible with this function to resample/reslice from isotropic
    voxel size to anisotropic or from isotropic to isotropic or even from
    anisotropic to anisotropic, as long as you provide the correct zooms
    (voxel sizes) and new_zooms (new voxel sizes). It is fairly easy to get the
    correct zooms using nibabel as show in the example below.

    Examples
    --------
    >>> import nibabel as nib
    >>> from dipy.align.aniso2iso import resample
    >>> from dipy.data import get_data
    >>> fimg=get_data('aniso_vox')
    >>> img=nib.load(fimg)
    >>> data=img.get_data()
    >>> data.shape
    (58, 58, 24)
    >>> affine=img.get_affine()
    >>> zooms=img.get_header().get_zooms()[:3]
    >>> zooms
    (4.0, 4.0, 5.0)
    >>> new_zooms=(3.,3.,3.)
    >>> new_zooms
    (3.0, 3.0, 3.0)
    >>> data2,affine2=resample(data,zooms,new_zooms,affine)
    >>> data2.shape
    (77, 77, 40)
    '''

    R = np.diag(np.array(new_zooms) / np.array(zooms))

    idx_spatial = list(np.arange(data.ndim))
    if data.ndim == 4:
        idx_spatial.remove(idx_spatial[time_axis])


    new_shape = np.array(zooms) / np.array(new_zooms) * \
        np.array(tuple(np.asarray(data.shape)[idx_spatial]))
    new_shape = np.round(new_shape).astype('i8')
    if data.ndim == 3:
        if not np.iscomplexobj(data):
            data2 = affine_transform(input=data, matrix=R,
                                     offset=np.zeros(3,),
                                     output_shape=tuple(new_shape),
                                     order=order, mode=mode, cval=cval)
        else:
            data2 = affine_transform(input=data.real, matrix=R,
                                     offset=np.zeros(3,),
                                     output_shape=tuple(new_shape),
                                     order=order, mode=mode, cval=cval) + \
                1j * affine_transform(input=data.imag, matrix=R,
                                      offset=np.zeros(3,),
                                      output_shape=tuple(new_shape),
                                      order=order, mode=mode, cval=cval)
    if data.ndim == 4:
        data2l = []
        slices = [slice(None)] * 4
        for i in range(data.shape[time_axis]):
            slices[time_axis] = i
            if not np.iscomplexobj(data):
                tmp = affine_transform(input=data[slices], matrix=R,
                                       offset=np.zeros(3,),
                                       output_shape=tuple(new_shape),
                                       order=order, mode=mode, cval=cval)
            else:
                tmp = affine_transform(input=data[slices].real,
                                       matrix=R, offset=np.zeros(3,),
                                       output_shape=tuple(new_shape),
                                       order=order, mode=mode, cval=cval) + \
                    +1j * affine_transform(input=data[slices].imag,
                                           matrix=R, offset=np.zeros(3,),
                                           output_shape=tuple(new_shape),
                                           order=order, mode=mode, cval=cval)
            data2l.append(tmp)
        data2_shape = np.asarray(data.shape)
        data2_shape[idx_spatial] = new_shape
        data2 = np.zeros(tuple(data2_shape), data.dtype)
        for i in range(data.shape[time_axis]):
            slices[time_axis] = i
            data2[slices] = data2l[i]
    if affine is not None:
        Rx = np.eye(4)
        Rx[:3, :3] = R
        affine2 = np.dot(affine, Rx)
        return data2, affine2
    else:
        return data2


def array_to_iso(data, zooms, new_zooms=None, affine=None, order=0,
                 mode='constant', cval=0):
    zooms = list(zooms)

    if new_zooms is None:
        new_zooms = [np.min(zooms), ] * len(zooms)
    else:
        new_zooms = list(new_zooms)

    if affine is None:
        data = resample_image(data, zooms, new_zooms, affine, order=0,
                              mode=mode, cval=cval)
        return data
    else:
        data, new_affine = resample_image(data, zooms, new_zooms, affine,
                                          order=0, mode=mode, cval=cval)
        return data, new_affine
