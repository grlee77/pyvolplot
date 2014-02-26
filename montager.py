# -*- coding: utf-8 -*-
""" 
**Create Image Montage**
"""
  
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np

#import sys
#if sys.hexversion > 0x03000000

def montager(xi, col=None, row=None, aspect=1.4, transpose=False, isRGB=False, flipx=False, flipy=False, flipz=False):
    """ tile a 3D or 4D image into a single 2D montage
    
    Parameters
    ----------
    xi : ndarray
        image data to montage
    col : int, optional
        number of columns in the montage
    row : int, optional
        number of rows in the montage
    aspect : float, optional
        desired aspect ratio of the montage
    transpose : bool, optional
        transpose each image slice in the montage? (transposes first two dimensions of the input)
    isRGB : bool, optional
        set True if the input is RGB
    flipx : bool, optional
        reverse x-axis indices?
    flipy : bool, optional
        reverse y-axis indices?
    flipz : bool, optional
        reverse z-axis indices?
    
    Returns
    -------
    xo : ndarray
        2D ndarray containing the montage
        
    Notes
    -----
    Any axis flips are applied prior to transposition    
    adapted from:  montager.m  from the image reconstruction toolbox by Jeff Fessler
    added RGB support, aspect ratio, transpose flag and axis flip flags

    
    """
    
    if isRGB: #call montager for R,G,B channels separately
        if xi.shape[-1]!=3:
            raise Exception("if isRGB=True, the last dimension must be size 3")
        if xi.ndim==5:
            xiR=xi[:,:,:,:,0]
            xiG=xi[:,:,:,:,1]
            xiB=xi[:,:,:,:,2]
        elif xi.ndim==4:
            xiR=xi[:,:,:,0]
            xiG=xi[:,:,:,1]
            xiB=xi[:,:,:,2]
        elif xi.ndim==3:
            xiR=xi[:,:,0]
            xiG=xi[:,:,1]
            xiB=xi[:,:,2]
        else:
            raise Exception("RGB inputs with xi.ndim>5 not supported")
        xoR=montager(xiR, col=col, row=row, aspect=aspect, transpose=transpose, isRGB=False, flipx=flipx, flipy=flipy, flipz=flipz)
        xoR=xoR.reshape(xoR.shape[0],xoR.shape[1],1)
        xoG=montager(xiG, col=col, row=row, aspect=aspect, transpose=transpose, isRGB=False, flipx=flipx, flipy=flipy, flipz=flipz)
        xoG=xoG.reshape(xoG.shape[0],xoG.shape[1],1)
        xoB=montager(xiB, col=col, row=row, aspect=aspect, transpose=transpose, isRGB=False, flipx=flipx, flipy=flipy, flipz=flipz)
        xoB=xoB.reshape(xoB.shape[0],xoB.shape[1],1)
        xo=np.concatenate((xoR,xoG,xoB),axis=2)
        return xo
        
    if xi is None or xi =='test':
        montager_test()
        return
    
    if xi.ndim > 4:
        print('ERROR in %s: >4D not done' % __name__)
    if xi.ndim == 4:
        if flipx:
            xi=xi[::-1,:,:,:]
        if flipy:
            xi=xi[:,::-1,:,:]
        if flipz:
            xi=xi[:,:,::-1,:]                                 
    	if transpose:
            xi=np.transpose(xi,axes=(1,0,2,3))   
        (nx, ny, n3, n4) = xi.shape;
    	nz = n3*n4;
    	#xi = reshape(xi, [nx ny nz]);
        xi = np.reshape(xi, [nx, ny, nz],order='F');
    elif xi.ndim ==3:
        if flipx:
            xi=xi[::-1,:,:]
        if flipy:
            xi=xi[:,::-1,:]
        if flipz:
            xi=xi[:,:,::-1]                                 
    	if transpose:
            xi=np.transpose(xi,axes=(1,0,2))            
    	[nx, ny, nz] = xi.shape;
    else: #for 1D or 2D case, just return the input, unchanged
        if flipx:
            xi=xi[::-1,:]
        if flipy:
            xi=xi[:,::-1]
        if transpose:
            xi=xi.T
        return xi
    
    if not col:
    	if not row:
    		if xi.ndim == 4:
    			col = n3;
    		#elif nx == ny and nz == np.round(np.sqrt(nz))**2: # perfect square
    		#	col = np.round(np.sqrt(nz))
    		else:
    			col = np.round(np.sqrt(nz * ny / nx * aspect));
    	else:
    		col = np.ceil(nz / row);
    
    if not row:
    	row = np.ceil(nz / col);
    
    xo = np.zeros((ny * row,nx * col));
    
    for iz in range(0,nz):
    	iy = np.floor(iz / col);
    	ix = iz - iy * col;
    	xo[iy*ny:(iy+1)*ny,ix*nx:(ix+1)*nx] = xi[:,:,iz].T
            
    return xo
    
def montager_test():
    t = (20, 30, 5);
    t = np.reshape(np.arange(0,np.prod(t)), t,order='F');  #order='F' to match matlab behavior
    plt.figure()
    plt.subplot(121)
    plt.imshow(montager(t),cmap=plt.cm.gray,interpolation='nearest')
    plt.subplot(122)
    plt.imshow(montager(t,row=1),cmap=plt.cm.gray,interpolation='nearest')

    
