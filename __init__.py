# -*- coding: utf-8 -*-
import matplotlib as mpl
from aniso2iso import array_to_iso, resample_image
from montager import montager, montager4d
from subplot_stack import subplot_stack
from centerplanes_stack import centerplanes_stack, centerplanes_stack_RGB
from mips import plot_mips

#modify some default matplotlib behavior
mpl.rcParams['image.cmap']='gray'
mpl.rcParams['image.interpolation']='nearest'
#mpl.rcParams['text.usetex']=True

#mpl.rcParams['axes.grid']=False
#mpl.rcParams['axes.color_cycle', ['b', 'g', 'r', 'c', 'm', 'y', 'k']]


"""
matplotlib.rcParamsDefault for reference:

RcParams({'agg.path.chunksize': 0,
          'animation.avconv_args': '',
          'animation.avconv_path': 'avconv',
          'animation.bitrate': -1,
          'animation.codec': 'mpeg4',
          'animation.convert_args': '',
          'animation.convert_path': 'convert',
          'animation.ffmpeg_args': '',
          'animation.ffmpeg_path': 'ffmpeg',
          'animation.frame_format': 'png',
          'animation.mencoder_args': '',
          'animation.mencoder_path': 'mencoder',
          'animation.writer': 'ffmpeg',
          'axes.axisbelow': False,
          'axes.color_cycle': ['b', 'g', 'r', 'c', 'm', 'y', 'k'],
          'axes.edgecolor': 'k',
          'axes.facecolor': 'w',
          'axes.formatter.limits': [-7, 7],
          'axes.formatter.use_locale': False,
          'axes.formatter.use_mathtext': False,
          'axes.grid': False,
          'axes.hold': True,
          'axes.labelcolor': 'k',
          'axes.labelsize': 'medium',
          'axes.labelweight': 'normal',
          'axes.linewidth': 1.0,
          'axes.titlesize': 'large',
          'axes.unicode_minus': True,
          'axes.xmargin': 0,
          'axes.ymargin': 0,
          'axes3d.grid': True,
          'backend': 'Agg',
          'backend.qt4': 'PyQt4',
          'backend_fallback': True,
          'contour.negative_linestyle': 'dashed',
          'datapath': '/home/lee8rx/anaconda/lib/python2.7/site-packages/matplotlib/mpl-data',
          'docstring.hardcopy': False,
          'examples.directory': '',
          'figure.autolayout': False,
          'figure.dpi': 80,
          'figure.edgecolor': 'w',
          'figure.facecolor': '0.75',
          'figure.figsize': [8.0, 6.0],
          'figure.frameon': True,
          'figure.max_open_warning': 20,
          'figure.subplot.bottom': 0.1,
          'figure.subplot.hspace': 0.2,
          'figure.subplot.left': 0.125,
          'figure.subplot.right': 0.9,
          'figure.subplot.top': 0.9,
          'figure.subplot.wspace': 0.2,
          'font.cursive': ['Apple Chancery',
                           'Textile',
                           'Zapf Chancery',
                           'Sand',
                           'cursive'],
          'font.family': 'sans-serif',
          'font.fantasy': ['Comic Sans MS',
                           'Chicago',
                           'Charcoal',
                           'ImpactWestern',
                           'fantasy'],
          'font.monospace': ['Bitstream Vera Sans Mono',
                             'DejaVu Sans Mono',
                             'Andale Mono',
                             'Nimbus Mono L',
                             'Courier New',
                             'Courier',
                             'Fixed',
                             'Terminal',
                             'monospace'],
          'font.sans-serif': ['Bitstream Vera Sans',
                              'DejaVu Sans',
                              'Lucida Grande',
                              'Verdana',
                              'Geneva',
                              'Lucid',
                              'Arial',
                              'Helvetica',
                              'Avant Garde',
                              'sans-serif'],
          'font.serif': ['Bitstream Vera Serif',
                         'DejaVu Serif',
                         'New Century Schoolbook',
                         'Century Schoolbook L',
                         'Utopia',
                         'ITC Bookman',
                         'Bookman',
                         'Nimbus Roman No9 L',
                         'Times New Roman',
                         'Times',
                         'Palatino',
                         'Charter',
                         'serif'],
          'font.size': 12,
          'font.stretch': 'normal',
          'font.style': 'normal',
          'font.variant': 'normal',
          'font.weight': 'normal',
          'grid.alpha': 1.0,
          'grid.color': 'k',
          'grid.linestyle': ':',
          'grid.linewidth': 0.5,
          'image.aspect': 'equal',
          'image.cmap': 'jet',
          'image.interpolation': 'bilinear',
          'image.lut': 256,
          'image.origin': 'upper',
          'image.resample': False,
          'interactive': False,
          'keymap.all_axes': 'a',
          'keymap.back': ['left', 'c', 'backspace'],
          'keymap.forward': ['right', 'v'],
          'keymap.fullscreen': ('f', 'ctrl+f'),
          'keymap.grid': 'g',
          'keymap.home': ['h', 'r', 'home'],
          'keymap.pan': 'p',
          'keymap.quit': ('ctrl+w', 'cmd+w'),
          'keymap.save': ('s', 'ctrl+s'),
          'keymap.xscale': ['k', 'L'],
          'keymap.yscale': 'l',
          'keymap.zoom': 'o',
          'legend.borderaxespad': 0.5,
          'legend.borderpad': 0.4,
          'legend.columnspacing': 2.0,
          'legend.fancybox': False,
          'legend.fontsize': 'large',
          'legend.frameon': True,
          'legend.handleheight': 0.7,
          'legend.handlelength': 2.0,
          'legend.handletextpad': 0.8,
          'legend.isaxes': True,
          'legend.labelspacing': 0.5,
          'legend.loc': 'upper right',
          'legend.markerscale': 1.0,
          'legend.numpoints': 2,
          'legend.scatterpoints': 3,
          'legend.shadow': False,
          'lines.antialiased': True,
          'lines.color': 'b',
          'lines.dash_capstyle': 'butt',
          'lines.dash_joinstyle': 'round',
          'lines.linestyle': '-',
          'lines.linewidth': 1.0,
          'lines.marker': 'None',
          'lines.markeredgewidth': 0.5,
          'lines.markersize': 6,
          'lines.solid_capstyle': 'projecting',
          'lines.solid_joinstyle': 'round',
          'mathtext.bf': 'serif:bold',
          'mathtext.cal': 'cursive',
          'mathtext.default': 'it',
          'mathtext.fallback_to_cm': True,
          'mathtext.fontset': 'cm',
          'mathtext.it': 'serif:italic',
          'mathtext.rm': 'serif',
          'mathtext.sf': 'sans\\-serif',
          'mathtext.tt': 'monospace',
          'patch.antialiased': True,
          'patch.edgecolor': 'k',
          'patch.facecolor': 'b',
          'patch.linewidth': 1.0,
          'path.effects': [],
          'path.simplify': True,
          'path.simplify_threshold': 0.1111111111111111,
          'path.sketch': None,
          'path.snap': True,
          'pdf.compression': 6,
          'pdf.fonttype': 3,
          'pdf.inheritcolor': False,
          'pdf.use14corefonts': False,
          'pgf.debug': False,
          'pgf.preamble': [''],
          'pgf.rcfonts': True,
          'pgf.texsystem': 'xelatex',
          'plugins.directory': '.matplotlib_plugins',
          'polaraxes.grid': True,
          'ps.distiller.res': 6000,
          'ps.fonttype': 3,
          'ps.papersize': 'letter',
          'ps.useafm': False,
          'ps.usedistiller': False,
          'savefig.bbox': None,
          'savefig.directory': '~',
          'savefig.dpi': 100,
          'savefig.edgecolor': 'w',
          'savefig.extension': 'png',
          'savefig.facecolor': 'w',
          'savefig.format': 'png',
          'savefig.frameon': True,
          'savefig.jpeg_quality': 95,
          'savefig.orientation': 'portrait',
          'savefig.pad_inches': 0.1,
          'svg.embed_char_paths': True,
          'svg.fonttype': 'path',
          'svg.image_inline': True,
          'svg.image_noscale': False,
          'text.antialiased': True,
          'text.color': 'k',
          'text.dvipnghack': None,
          'text.hinting': True,
          'text.hinting_factor': 8,
          'text.latex.preamble': [''],
          'text.latex.preview': False,
          'text.latex.unicode': False,
          'text.usetex': False,
          'timezone': 'UTC',
          'tk.pythoninspect': False,
          'tk.window_focus': False,
          'toolbar': 'toolbar2',
          'verbose.fileo': 'sys.stdout',
          'verbose.level': 'silent',
          'webagg.open_in_browser': True,
          'webagg.port': 8988,
          'webagg.port_retries': 50,
          'xtick.color': 'k',
          'xtick.direction': 'in',
          'xtick.labelsize': 'medium',
          'xtick.major.pad': 4,
          'xtick.major.size': 4,
          'xtick.major.width': 0.5,
          'xtick.minor.pad': 4,
          'xtick.minor.size': 2,
          'xtick.minor.width': 0.5,
          'ytick.color': 'k',
          'ytick.direction': 'in',
          'ytick.labelsize': 'medium',
          'ytick.major.pad': 4,
          'ytick.major.size': 4,
          'ytick.major.width': 0.5,
          'ytick.minor.pad': 4,
          'ytick.minor.size': 2,
          'ytick.minor.width': 0.5})

"""


