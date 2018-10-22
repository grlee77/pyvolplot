# from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


def is_string_like(obj):
    """Check if obj is string."""
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


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
