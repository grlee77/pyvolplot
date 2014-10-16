# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import pywt


def plot_wavelet_coeffs(
        coeffs, generate_figures=True, open_newfig=True, wavelet='db8'):

    if not isinstance(wavelet, pywt.Wavelet):
        wavelet = pywt.Wavelet('db8')

    def norm_diff(d):
        d = d - d.min()
        return d / d.max()

    # coeffs=coeffs[-1::-1]
    aa = coeffs[0]
    coeffs = coeffs[1:]

    idx_start = wavelet.rec_len / 2 - 1
    idx_stop = -wavelet.rec_len / 2

    if len(coeffs) > 1:
        for coeff in coeffs[:-1]:
            tmp1 = np.concatenate((norm_diff(aa),
                                   norm_diff(coeff['ad'])), axis=0)
            tmp2 = np.concatenate((norm_diff(coeff['da']),
                                   norm_diff(coeff['dd'])), axis=0)
            tmp1 = np.concatenate((tmp1, tmp2), axis=1)
            aa = tmp1[idx_start:idx_stop, idx_start:idx_stop]
        coeffs = coeffs[-1]
    else:
        coeffs = coeffs[0]

    if ('ad' not in coeffs) or ('dd' not in coeffs) or ('da' not in coeffs):
        raise ValueError(
            "Invalid coeffs dictionary.  Expected to find keys 'ad','da','dd'")

    tmp1 = np.concatenate((norm_diff(aa), norm_diff(coeffs['ad'])), axis=0)
    tmp2 = np.concatenate((norm_diff(coeffs['da']),
                           norm_diff(coeffs['dd'])), axis=0)
    tmp1 = np.concatenate((tmp1, tmp2), axis=1)
    aa = tmp1[idx_start:idx_stop, idx_start:idx_stop]

    if generate_figures:
        if open_newfig:
            plt.figure()
        plt.imshow(aa)

    return aa
