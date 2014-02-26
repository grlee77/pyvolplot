import numpy as np
from skimage import img_as_float
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import BaseWidget, Slider, OKCancelButtons, CheckBox, SaveButtons, ComboBox

#from skimage.viewer.plugins import PlotPlugin

from skimage.viewer.qt import QtGui, QtCore
from skimage.viewer.qt.QtCore import Qt, Signal

from skimage import data
from PyIRT.denoising.nlmeans import nlmeans
from PyIRT.denoising.utils import add_noise
from skimage.viewer.plugins import Plugin, PlotPlugin


from skimage.viewer.viewers.core import BlitManager, EventManager
from skimage.viewer import utils

# class MultiImageViewer(ImageViewer):
#     def __init__(self, images, addnoise_widget = True, useblit=True):
#         first_image = image_collection[images]
#         super(MultiImageViewer, self).__init__(first_image, useblit=useblit)

        
class DenoiseImageViewer(ImageViewer):

    def __init__(self, image, oracle_image = None, addnoise_widget = True, useblit=True):
#        super(DenoiseImageViewer, self).__init__(image, useblit=useblit)
        
        # self.layout = QtGui.QVBoxLayout(self.main_widget)
        # self.fig_layout = QtGui.QHBoxLayout(self.mainwidget)
        # self.fig_layout.addWidget(self.canvas)

        # image_noisy = image.copy()
        # self.fig_noisy, self.ax_noisy = utils.figimage(image_noisy)
        # self.canvas_noisy = self.fig_noisy.canvas
        # self.canvas_noisy.setParent(self)   
        # self.ax_noisy.autoscale(enable=False)     
        # if useblit:
        #     self._blit_manager_noisy = BlitManager(self.ax_noisy)
        # self._event_manager_noisy = EventManager(self.ax_noisy)
        # self._noisy_image_plot = self.ax_noisy.images[0]
        # self.fig_layout.addWidget(self.canvas_noisy)
        # #self.layout.addWidget(self.canvas_noisy)

        # self.layout.addLayout(self.fig_layout)
        # sb_size = self.statusBar().sizeHint()
        # cs_size = self.canvas.sizeHint()
        # cs_size_noisy = self.canvas_noisy.sizeHint()
        # self.resize(cs_size.width() + cs_size_noisy.width(), 
        #             cs_size.height() + sb_size.height())


   # def __init__(self, image, useblit=True):
        # Start main loop
        utils.init_qtapp()
        super(ImageViewer, self).__init__()

        #TODO: Add ImageViewer to skimage.io window manager

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Image Viewer")

        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('Open file', self.open_file,
                                 Qt.CTRL + Qt.Key_O)
        self.file_menu.addAction('Save to file', self.save_to_file,
                                 Qt.CTRL + Qt.Key_S)
        self.file_menu.addAction('Quit', self.close,
                                 Qt.CTRL + Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.main_widget = QtGui.QWidget()
        self.setCentralWidget(self.main_widget)

        if isinstance(image, Plugin):
            plugin = image
            image = plugin.filtered_image
            plugin.image_changed.connect(self._update_original_image)
            # When plugin is started, start
            plugin._started.connect(self._show)

        self.fig, self.ax = utils.figimage(image)
        self.canvas = self.fig.canvas
        self.canvas.setParent(self)
        self.ax.autoscale(enable=False)

        self._tools = []
        self.useblit = useblit
        if useblit:
            self._blit_manager = BlitManager(self.ax)
        self._event_manager = EventManager(self.ax)

        self._image_plot = self.ax.images[0]
        self._update_original_image(image)
        self.plugins = []

        self.layout = QtGui.QVBoxLayout(self.main_widget)

        self.fig_layout = QtGui.QHBoxLayout(self.main_widget)
        self.fig_layout.addWidget(self.canvas)

        status_bar = self.statusBar()
        self.status_message = status_bar.showMessage
        sb_size = status_bar.sizeHint()
        cs_size = self.canvas.sizeHint()
        cs_width = cs_size.width()

        image_noisy = image.copy()
        self.fig_noisy, self.ax_noisy = utils.figimage(image_noisy)
        self.canvas_noisy = self.fig_noisy.canvas
        self.canvas_noisy.setParent(self)   
        self.ax_noisy.autoscale(enable=False)     
        if useblit:
            self._blit_manager_noisy = BlitManager(self.ax_noisy)
        self._event_manager_noisy = EventManager(self.ax_noisy)
        self._noisy_image_plot = self.ax_noisy.images[0]
        self.fig_layout.addWidget(self.canvas_noisy)
        cs_size_noisy = self.canvas_noisy.sizeHint()
        cs_width += cs_size_noisy.width()

        if oracle_image is not None:
            image_oracle = oracle_image.copy()
            self.fig_oracle, self.ax_oracle = utils.figimage(image_oracle)
            self.canvas_oracle = self.fig_oracle.canvas
            self.canvas_oracle.setParent(self)   
            self.ax_oracle.autoscale(enable=False)     
            if useblit:
                self._blit_manager_oracle = BlitManager(self.ax_oracle)
            self._event_manager_oracle = EventManager(self.ax_oracle)
            self._oracle_image_plot = self.ax_oracle.images[0]
            self.fig_layout.addWidget(self.canvas_oracle)        
            cs_size_oracle = self.canvas_oracle.sizeHint()
            cs_width += cs_size_oracle.width()

        self.layout.addLayout(self.fig_layout)


        self.resize(cs_width, cs_size.height() + sb_size.height())

        self.connect_event('motion_notify_event', self._update_status_bar)


    @property
    def noisy_image(self):
        return self._noisy_img

    @noisy_image.setter
    def noisy_image(self, noisy_image):
        self._noisy_img = noisy_image
        utils.update_axes_image(self._noisy_image_plot, noisy_image)

        # update display (otherwise image doesn't fill the canvas)
        h, w = noisy_image.shape[:2]
        self.ax_noisy.set_xlim(0, w)
        self.ax_noisy.set_ylim(h, 0)

        # update color range
        clim = dtype_range[image.dtype.type]
        if clim[0] < 0 and image.min() >= 0:
            clim = (0, clim[1])
        self._noisy_image_plot.set_clim(clim)

        if self.useblit:
            self._blit_manager_noisy.background = None

        self.redraw_noisy()

    def redraw_noisy(self):
        if self.useblit:
            self._blit_manager_noisy.redraw()
        else:
            self.canvas_noisy.draw_idle()



#plugin = PlotPlugin(image_filter=nlmeans) # doctest: +SKIP
plugin = Plugin(image_filter=nlmeans) # doctest: +SKIP

plugin += Slider('patch_radius', 0, 5, value = 1, value_type='int')
plugin += Slider('block_radius', 0, 30, value = 5, value_type='int')
plugin += Slider('sigma', 0.001, 1, value = 0.1, value_type='float')
plugin += Slider('h', 0.001, 1, value = 0.1, value_type='float')
plugin += CheckBox(name='use_threshold_images', value = False)
plugin += Slider('mu1', 0.001, 1, value = 0.95, value_type='float')
plugin += Slider('sig1', 0.001, 1, value = 0.5, value_type='float')
plugin += CheckBox(name='rician', value = False)
plugin += CheckBox(name='estimate_local_std', value = False)
plugin += Slider('std_estimate_scale', 0.0, 3.0, value = 1.0, value_type='float')
plugin += Slider('est_patch_radius', 0, 5, value = 1, value_type='int')
plugin += Slider('est_block_radius', 0, 30, value = 5, value_type='int')

#plugin += Slider('h', 0, 100, value_type='float')
#plugin += Slider('threshold', 0, 255)       # doctest: +SKIP

image = img_as_float(data.coins())
sigma = 30/255.

image_noisy = image + np.random.normal(loc=0, scale=sigma, size=image.shape)
image_noisy = np.clip(image_noisy, 0, 1)

# image = data.coins()
# noise = np.random.random(image.shape)
# sigma = 50
# image_noisy = add_noise(image, sigma=30, noise_type='Gaussian', generate_figures=True)
# image_noisy /= 255

#image_noisy = np.clip(image_noisy/255,0,1)
viewer = DenoiseImageViewer(image_noisy,oracle_image=data.coins())       # doctest: +SKIP
#viewer.show()
plugin += OKCancelButtons()

viewer += plugin
filtered = viewer.show() #[0][0] # doctest: +SKIP

# arr, sigma = None, h = None, patch_radius = None, 
#             block_radius = None, use_threshold_images = True, 
#             mu1 = 0.95, sig1 = 0.5, normalize = False, rician = False, 
#             mask = None, z = None,
#             generate_figures = False,enable_4D = False,kernelv2 = True,
#             use_oracle = False, oracle_arr = None, estimate_local_std = False,
#             std_estimate_scale = 1.0,est_patch_radius = None, est_block_radius = None,
#             loop_axis = None, sigma_arr = None