import numpy as np
from matplotlib import pyplot as plt
from pyvolplot import volshow
from pyvolplot.overlay import masked_overlay
from pyvolplot.montager import montager


def test_overlay():
    """ make a crude CSF mask, to test the overlay function """
    b = np.load('T1_recovery_lowres_4D.npz')['T1vols']
    mask1 = b[..., 0] > 0.05 * b[..., 0].max()
    mask2 = b[..., 7] < 0.3 * b[..., 7].max()
    csf_mask = mask1 & mask2
    volshow(b[..., 7], mode='m', transpose=True)
    masked_overlay(montager(csf_mask, transpose=True), ax=plt.gca(),
    			   vmax=2.5, alpha=1.0, maskval=0, add_colorbar=False)
