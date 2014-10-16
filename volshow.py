# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import inspect
import warnings
import numpy as np
from matplotlib import pyplot as plt

from .montager import montager, montager4d, add_lines
from .centerplanes_stack import centerplanes_stack, centerplanes_stack_RGB
from .mips import calc_mips
from PyIRT.utils import is_string_like

def masked_overlay(image, overlay_cmap=plt.cm.hot, vmin=None, vmax=None, 
                   alpha_image = None, maskval=0): #, f=None):
                       
    """ overlay another volume via alpha transparency """
    if vmin is None:
        vmin = image.min()
    if vmax is None:
        vmax = image.max() 
    if alpha_image is None:
        if maskval is not None:
            alpha_mask = (image==maskval)
    else:
        alpha_mask = np.ones_like(alpha_image)
    image = (np.clip(image,vmin,vmax)-vmin)/(vmax-vmin)
    image_RGBA = overlay_cmap(image)  #convert to RGBA
    if alpha_mask is not None:
        if alpha_image is None:
            image_RGBA[...,-1][alpha_mask] = 0  #set
        else:
            image_RGBA[...,-1] = image_RGBA[...,-1]*alpha_image
#    if f is None:
#        f = plt.figure()
    plt.imshow(image_RGBA,cmap=overlay_cmap,vmin=vmin,vmax=vmax)
    plt.colorbar(shrink=0.9)
    plt.axis('off')
    plt.axis('image')
    return
#    return f
    
def volshow(x,ax=None,mode=None,cplx_to_abs=True,show_lines=False,line_color='w',mask_nan=True,**kwargs):
    #TODO: add overlay volumes

    x=np.asanyarray(x)
    if np.iscomplexobj(x):
        if cplx_to_abs:
            print("magnitude of complex data displayed")
            x=np.abs(x)
        else:
            print("only real part of complex data displayed")
            x=np.real(x)
    
    if mask_nan:
        nanmask = np.isnan(x)
        if np.any(nanmask):
            print("NaN values found... setting them to 0 for display")
        x[nanmask] = 0

    #if input is RGB or RGBA reduce number of dimensions to tile by 1
    isRGB = kwargs.get('isRGB',False)
    if (x.ndim >= 3) and isRGB:
        nd = x.ndim-1
    else:
        nd = x.ndim

    if nd < 2:
        warnings.warn("input only has {} dimensions. converting to 2D".format(nd))
        x = np.atleast_2d(x)

    #generate a new figure if an existing axis was not passed in
    if ax is None:
        f=plt.figure()
        ax=f.add_subplot(1,1,1)

    if mode is None:
        if nd >=3:
            mode = 'montage'
        else:
            mode = 'imshow'
            
    if not is_string_like(mode):
        raise ValueError("mode must be string-like")
        
    print("mode={}".format(mode))
    if mode.lower() in ['m', 'montage']: 
        print("montage mode")
        if nd>4:
            warnings.warn("montager only tiles up to 4D.  all excess dimensions collapsed into 4th (Fortran order)")
            x = x.reshape(x.shape[:3]+(-1,),order='F')
            
        #separate out options to the montager
        montage_args = []
        if nd==3:
            #list of keyword argument names for montager
            montage_args += inspect.getargspec(montager)[0]
        elif nd>3:  
            #4d montager has additional options
            montage_args += inspect.getargspec(montager4d)[0]
        
        montage_kwargs={}
        for k in montage_args:
            if k in kwargs:
                montage_kwargs[k]=kwargs.pop(k)
                
        montage_kwargs['xi'] = x
        montage_kwargs['output_grid_size'] = True
        #assume remainder of kwargs go to imshow    
        if nd>=4:
            x, nrows, ncols=montager4d(**montage_kwargs)
        else:
            x, nrows, ncols=montager(**montage_kwargs)

    elif mode.lower() in ['c', 'centerplanes']:
        print("centerplanes mode")
        if nd!=3:
            raise ValueError("centerplanes mode only supports 3D volumes")
            
        if isRGB:
            stack_func = centerplanes_stack_RGB
        else:
            stack_func = centerplanes_stack

        centerplanes_arg_list = inspect.getargspec(stack_func)[0]
        centerplane_kwargs={}
        for k in centerplanes_arg_list:
            if k in kwargs:
                centerplane_kwargs[k]=kwargs.pop(k)
        centerplane_kwargs['x'] = x
                
        x = stack_func(**centerplane_kwargs)
        
    elif mode.lower() in ['p','mips']:
        print("MIPS mode")
        if isRGB:
            raise ValueError("RGB not support for MIP mode")
        
        mips_arg_list = inspect.getargspec(plot_mips)[0]
        mip_kwargs={}
        for k in mips_arg_list:
            if k in kwargs:
                mip_kwargs[k]=kwargs.pop(k)
        mip_kwargs['x'] = x
        x = calc_mips(**mip_kwargs)
                
    elif mode.lower() in ['i','imshow']:
        print("imshow mode")
        if nd>2:
            raise ValueError("imshow mode only works for 2D input")
    else:
        raise ValueError("unsupported mode")

    #change a few of the imshow defaults
    if 'cmap' not in kwargs:
        kwargs['cmap']=plt.cm.gray
    if 'interpolation' not in kwargs:
        kwargs['interpolation']='nearest'
        
    ax.imshow(x,**kwargs)
    
    #if requested, draw lines dividing the cells of the montage
    if show_lines and (mode.lower() in ['m','montage']):
        #TODO: add lines for MIP or centerplane cases
        add_lines(x, color=line_color, ncells_horizontal=ncols, ncells_vertical=nrows)
        
    ax.axis('off')
    ax.axis('image')


 
        


    

        