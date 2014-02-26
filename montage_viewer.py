# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

from skimage.viewer import ImageViewer
from skimage import data, img_as_float

from skimage.viewer.plugins.base import Plugin
from skimage.viewer.plugins.plotplugin import PlotPlugin
from skimage.viewer.widgets import Slider
from skimage.exposure import rescale_intensity
from skimage.transform import rotate

from PyIRT.graph.montager import montager
from PyIRT.denoising.utils import denoising_examples

import numpy as np
import matplotlib.pyplot as plt

#col=None, row=None, aspect=1.4, transpose=False, isRGB=False, flipx=False, flipy=False, flipz=False)
z, mask, sigma = denoising_examples(example='brain3D', downsample_factor=4)

class TiledImageViewer(ImageViewer):
    def __init__(self, image, **kwargs):

        #auto-rescale        
        if image.max()>1.0:
            if image.min() < 0:
                image = image-image.min()
            image = 1.0 * image/image.max()

        self.original_volume = image
        self.ntiles = image.shape[-1]
        
            
        image = montager(self.original_volume, aspect=1.4)
        super(TiledImageViewer, self).__init__(image, **kwargs)

        col_slider_kwds = dict(value=np.sqrt(self.ntiles), low=1, high=self.ntiles , update_on='release',
                           callback=self.update_cols, value_type='int')

        self.col_slider = Slider('columns', **col_slider_kwds)
        self.layout.addWidget(self.col_slider)

        row_slider_kwds = dict(value=np.ceil(self.ntiles/np.sqrt(self.ntiles)), low=1, high=self.ntiles , update_on='release',
                           callback=self.update_rows, value_type='int')

        self.row_slider = Slider('rows', **row_slider_kwds)
        self.layout.addWidget(self.row_slider)

        #TODO: add aspect, flips, slice range, slice skip, transpose, etc controls

        self.origin_image = image

    def update_cols(self, name, columns):
        self.image = montager(self.original_volume, aspect=1.4, col=columns)
        #TODO: update row slider position
        #self.row_slider.val = new_rows

    def update_rows(self, name, rows):
        self.image = montager(self.original_volume, aspect=1.4, row=rows)
        #TODO: update column slider position
        #self.col_slider.val = new_cols
        

#montager_plugin = Plugin(image_filter = montager)       
#plugin += Slider('col', low=0, high=10, value_type = 'int')
#plugin += Slider('row', low=0, high=10, value_type = 'int)
        
image = z
tile_viewer = TiledImageViewer(image)
tile_viewer.show()

viewer = ImageViewer(image)
rotated_viewer = RotatedImageViewer(image)

histogram = Histogram(viewer)
rotated_viewer += histogram
rotated_viewer.histogram = histogram

super(ImageViewer, viewer).show()
rotated_viewer.show()


#TODO: make a separate doc widget for all the controls as follows:

#    def __add__(self, plugin):
#        """Add plugin to ImageViewer"""
#        plugin.attach(self)
#        self.original_image_changed.connect(plugin._update_original_image)
#
#        if plugin.dock:
#            location = self.dock_areas[plugin.dock]
#            dock_location = Qt.DockWidgetArea(location)
#            dock = QtGui.QDockWidget()
#            dock.setWidget(plugin)
#            dock.setWindowTitle(plugin.name)
#            self.addDockWidget(dock_location, dock)
#
#            horiz = (self.dock_areas['left'], self.dock_areas['right'])
#            dimension = 'width' if location in horiz else 'height'
#            self._add_widget_size(plugin, dimension=dimension)
#
#        return self
#
#    def _add_widget_size(self, widget, dimension='width'):
#        widget_size = widget.sizeHint()
#        viewer_size = self.frameGeometry()
#
#        dx = dy = 0
#        if dimension == 'width':
#            dx = widget_size.width()
#        elif dimension == 'height':
#            dy = widget_size.height()
#
#        w = viewer_size.width()
#        h = viewer_size.height()
#        self.resize(w + dx, h + dy)


###PLUGIN CODE###
#    self.setWindowTitle(self.name)
#        self.layout = QtGui.QGridLayout(self)
#        self.resize(width, height)
#        self.row = 0
#    def add_widget(self, widget):
#        """Add widget to plugin.
#
#        Alternatively, Plugin's `__add__` method is overloaded to add widgets::
#
#            plugin += Widget(...)
#
#        Widgets can adjust required or optional arguments of filter function or
#        parameters for the plugin. This is specified by the Widget's `ptype'.
#        """
#        if widget.ptype == 'kwarg':
#            name = widget.name.replace(' ', '_')
#            self.keyword_arguments[name] = widget
#            widget.callback = self.filter_image
#        elif widget.ptype == 'arg':
#            self.arguments.append(widget)
#            widget.callback = self.filter_image
#        elif widget.ptype == 'plugin':
#            widget.callback = self.update_plugin
#        widget.plugin = self
#        self.layout.addWidget(widget, self.row, 0)
#        self.row += 1