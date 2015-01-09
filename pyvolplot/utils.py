# from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


def add_inner_title(ax, title, loc=9, size=None, prop=None, **kwargs):
    """  Add a title within an image using AnchoredText().

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axis on which to add the title
    title : str
        title String
    loc : str or int, optional
        location number:  e.g. 9 = 'upper center'
    size : str or int, optional
        Font size
    prop : dict, optional
        dictionary of other font properties.  if `size` is specified it will
        override prop['size']
    kwargs : dict, optional
        extra keyword arguments to pass on to AnchoredText

    Note
    ----
    recommend using this with ImageGrid to add titles onto the individual
    images within the grid

    """

    # from matplotlib.patheffects import withStroke
    if size is None:
        size = u'x-large'
        # size = plt.rcParams['legend.fontsize']
    if prop is None:
        prop = dict(size=size, color='w', weight='bold')
    else:
        prop['size'] = size
    at = AnchoredText(title, loc=loc, pad=0., borderpad=0.5,
                      frameon=False, prop=prop, **kwargs)
    ax.add_artist(at)
    # at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


def reorient_LPI(arr, affine):
    """
    experimental:
    return copy of array with axes reordered and/or flipped as appropriate
    to give a volume in LPI orientation
    """
    import numpy as np
    try:
        from nibabel.orientations import aff2axcodes
    except ImportError as e:
        print("nibabel is required to use the reorient_to_LPR utility")
        raise e

    affine = np.asarray(affine)
    if affine.shape != (4, 4):
        raise ValueError("affine should be a 4 x 4 array)")
    axcodes = aff2axcodes(affine)

    permute = np.zeros((3,))
    for idx, ax in enumerate(axcodes):
        if ax == 'L' or ax == 'R':
            permute[0] = idx
        if ax == 'A' or ax == 'P':
            permute[1] = idx
        if ax == 'I' or ax == 'S':
            permute[2] = idx
    permute = tuple(permute.tolist())
    arr = arr.transpose(permute)

    if 'R' in axcodes:
        arr = arr[::-1, ...]
    if 'A' in axcodes:
        arr = arr[:, ::-1, ...]
    if 'S' in axcodes:
        arr = arr[:, :, ::-1, ...]
    return arr


def trim_to_mask(arr, mask, maskval=0, pad_width=1):
    """ Trim array borders based on values in mask.

    For any borders where mask==maskaval, return copy of arr and mask with
    these borders removed.

    Parameters
    ----------
    arr : array_like
        nd array to trim
    mask : array_like
        mask array with shape matching arr
    maskval : float or int
        value in the mask to consider as background
    pad_width: int or tuple of int
        After trim, add back borders of the specified width

    Returns
    -------
    arr : array_like
        copy of arr with borders trimmed
    mask : array_like
        copy of mask with borders trimmed
    """
    import numpy as np
    arr = np.asarray(arr).copy()
    mask = np.asarray(mask)
    maskbool = mask != maskval
    if arr.shape != mask.shape:
        raise ValueError("shapes of arrays must match")
    slices = [slice(None), ]*mask.ndim
    for d in range(mask.ndim):
        axes_to_sum = range(mask.ndim)
        axes_to_sum.remove(d)
        idx_keep = maskbool.sum(axis=tuple(axes_to_sum)).nonzero()[0]
        slices[d] = slice(idx_keep[0], idx_keep[-1])
        mask = mask[slices]
        arr = arr[slices]
        slices[d] = slice(None)
    arr = np.pad(arr, pad_width, mode='constant', constant_values=maskval)
    mask = np.pad(mask, pad_width, mode='constant',
                  constant_values=maskval)
    return arr, mask


def generate_atlas_colormap(num_ROIs, prepend_bg=True, bgcolor=(0, 0, 0),
                            randomize=True, random_seed=None, show_plot=False,
                            lightness=0.6, saturation=0.7):
    """ generate a discrete colormap to use with brain atlas overlays, etc.

    The colormap values will have constant brightness and saturation, with
    hue varying across the color space.

    Parameters
    ----------
    num_ROIs : int
        the number of discrete colors in the map.
    prepend_bg : bool, optional
        if True, prepend the bgcolor to the colormap.  Intended for use as the
        background color
    bgcolor : tuple, optional
        color to use for the background
    randomize : bool, optional
        if True, randomize the order of colors (aside from the background)
    random_seed : int, optional
        seed to use during randomization
    show_plot : bool, optional
        if True, call seaborn.palplot to display the colormap

    """
    from matplotlib import colors
    try:
        from seaborn.palettes import hls_palette
        from seaborn.miscplot import palplot
    except ImportError as e:
        print("seaborn is currently required to use generate_atlas_colormap")
        raise e

    palette = hls_palette(num_ROIs, s=saturation, l=lightness)
    if prepend_bg:
        palette = [bgcolor] + palette

    if randomize:
        from random import shuffle, seed
        if random_seed is not None:
            seed(random_seed)
        shuffle(palette)

    cmap = colors.ListedColormap(palette)

    if show_plot:
        palplot(palette)
    return cmap


def _cmap_single_channel(channel, N, smin=0.2, smax=1.0):
    """ single channel RGB colormap with N intries with in channel in range
        [smin, smax]
    """
    import numpy as np
    from matplotlib import colors
    if smin < 0 or smax > 1 or smin > smax:
        raise ValueError("require: 0 <= smin < smax <= 1")
    N = int(N)
    if N < 1:
        raise ValueError("N must be > 1")
    rgb = np.zeros((N, 3))
    rgb[:, channel] = np.linspace(smin, smax, N)
    color = rgb
    return colors.ListedColormap(color)


def cmap_greens(N, smin=0.2, smax=1.0):
    """ green RGB colormap with N entries in range [smin, smax]
    """
    return _cmap_single_channel(1, N, smin, smax)


def cmap_reds(N, smin=0.2, smax=1.0):
    """ red RGB colormap with N intries with in range [smin, smax]
    """
    return _cmap_single_channel(0, N, smin, smax)


def cmap_blues(N, smin=0.2, smax=1.0):
    """ blue RGB colormap with N intries with in range [smin, smax]
    """
    return _cmap_single_channel(2, N, smin, smax)
