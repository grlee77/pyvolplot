from __future__ import division  #avoids integer division so that 3/2 = 1.5 and not 1 
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def montager(xi, col=[], row=[], aspect=1.2, transpose=False):
    """
    %function xo = montager(xi, varargin)
    % in
    %	xi	[3d or 4d] set of images
    % options
    %	'col'		# of cols
    %	'row'		# of rows
    %	'aspect'	aspect ratio (default 1.2)
    % out
    %	xo [2d]		3d or 4d images arrange a as a 2d rectangular montage
    % adapted from:  montager.m, 1997, Jeff Fessler, The University of Michigan
    """
    
    if xi is None or xi =='test':
        montager_test()
        return
    
    if xi.ndim > 4:
        print 'ERROR in %s: >4D not done' % __name__
    if xi.ndim == 4:
    	(nx, ny, n3, n4) = xi.shape;
    	nz = n3*n4;
    	#xi = reshape(xi, [nx ny nz]);
        xi = np.reshape(xi, [nx, ny, nz],order='F');
    else:
    	[nx, ny, nz] = xi.shape;
    
    if not col:
    	if not row:
    		if xi.ndim == 4:
    			col = n3;
    		elif nx == ny and nz == np.round(np.sqrt(nz))**2: # perfect square
    			col = np.round(np.sqrt(nz))
    		else:
    			col = np.ceil(np.sqrt(nz * ny / nx * aspect));
    	else:
    		col = np.ceil(nz / row);
    
    if not row:
    	row = np.ceil(nz / col);
    
    xo = np.zeros((nx * col, ny * row));
    
    for iz in range(0,nz):
    	iy = np.floor(iz / col);
    	ix = iz - iy * col;
    	#xo[np.arange(0,nx)+int(ix*nx), np.arange(0,ny)+int(iy*ny)] = xi[:,:,iz]
        if transpose:
            xo[iy*ny:(iy+1)*ny,ix*nx:(ix+1)*nx] = xi[:,:,iz].T
        else:
            xo[ix*nx:(ix+1)*nx,iy*ny:(iy+1)*ny] = xi[:,:,iz]
    return xo
    
def montager_test():
    t = (20, 30, 5);
    t = np.reshape(np.arange(0,np.prod(t)), t,order='F');  #order='F' to match matlab behavior
    plt.figure()
    plt.subplot(121)
    plt.imshow(montager(t).T,cmap=plt.cm.gray,interpolation='nearest')
    plt.subplot(122)
    plt.imshow(montager(t,row=4).T,cmap=plt.cm.gray,interpolation='nearest')

    
