# -*- coding: utf-8 -*-
""" 
**Create Image Montage**
"""
  
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np

def add_lines(e, color='w',ncells_horizontal=1, ncells_vertical=1):
    """ takes output from montager and the number of rows/cols and plots lines
    separating the cells """
    
    e=np.asanyarray(e)
    if e.ndim!=2:
        raise ValueError("Requires 2D array")
    if ncells_horizontal>1:
        s=e.shape[1]
        line_locs=np.round(np.arange(1,ncells_horizontal)*s/ncells_horizontal)
        for l in line_locs:
            plt.plot([l,l],[0,e.shape[0]],color);
            plt.axis('image')
    if ncells_horizontal>1:
        s=e.shape[0]
        line_locs=np.round(np.arange(1,ncells_vertical)*s/ncells_vertical)
        for l in line_locs:
            plt.plot([0,e.shape[1]],[l,l],color);
            plt.axis('image')            

def montager4d(xi,axis=-1,row2=None,col2=None,aspect2=None,**montager_args):
    """ nested montager for 4D data.
        montage each 3D subvolume, then montage the montages 
    """
    #TODO: support ndim=5 for RGBA input
    #TODO?: allow input < size 4, just pad size to ones on missing axes?
    xi=np.asanyarray(xi)
    if xi.ndim != 4:  
        raise ValueError("montager4d requires 4d input")
    if montager_args.get('isRGB',False):
        raise ValueError("isRGB=True not currently supported")
        
    nvols = xi.shape[axis]
    slices = [slice(None),]*4
    slices[axis] = 0
    m0 = montager(xi[slices],**montager_args)
    m_out = np.zeros(m0.shape+(nvols,),dtype=xi.dtype)
    m_out[:,:,0] = m0
    for n in range(1,nvols):
        slices[axis] = n
        m_out[:,:,n] = montager(xi[slices],**montager_args)
    montage2_args=montager_args.copy()
    #flip and transpose operations skipped on second time through montager
    montage2_args['transpose']=True
    montage2_args['flipx']=False
    montage2_args['flipy']=False
    montage2_args['flipz']=False
    montage2_args['isRGB']=False
    montage2_args['col']=col2
    montage2_args['row']=row2
    if aspect2 is not None:
        montage2_args['aspect']=aspect2
    return montager(m_out,**montage2_args)
        
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
    #TODO: (add linecolor arg).  if not None, plot lines separating cells
    #TODO?: also allow RGBA axis to be the first rather than last
    #TODO: add option to add a border between the cells
    #TODO: allow >4D by stacking all remaining dimensions along the 4th
    
    if isRGB: #call montager for R,G,B channels separately
        if xi.shape[-1]<3 or xi.shape[-1]>4:
            raise Exception("if isRGB=True, the last dimension must be size 3")
        if xi.shape[-1]==4:
            has_alpha = True
        else:
            has_alpha = False
        
        xiR=xi[...,0]
        xiG=xi[...,1]
        xiB=xi[...,2]
        xoR=montager(xiR, col=col, row=row, aspect=aspect, transpose=transpose, 
                     isRGB=False, flipx=flipx, flipy=flipy, flipz=flipz)
        xoR=xoR[:,:,None]
        xoG=montager(xiG, col=col, row=row, aspect=aspect, transpose=transpose, 
                     isRGB=False, flipx=flipx, flipy=flipy, flipz=flipz)
        xoG=xoG[:,:,None]
        xoB=montager(xiB, col=col, row=row, aspect=aspect, transpose=transpose, 
                     isRGB=False, flipx=flipx, flipy=flipy, flipz=flipz)
        xoB=xoB[:,:,None]
        if has_alpha:
            xiA=xi[...,3]    
            xoA=montager(xiA, col=col, row=row, aspect=aspect, 
                         transpose=transpose, isRGB=False, flipx=flipx, 
                         flipy=flipy, flipz=flipz)
            xoA=xoA[:,:,None]
            xo=np.concatenate((xoR,xoG,xoB,xoA),axis=2)
        else:
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
        xi = np.reshape(xi, (nx, ny, nz), order='F');
    elif xi.ndim ==3:
        if flipx:
            xi=xi[::-1,:,:]
        if flipy:
            xi=xi[:,::-1,:]
        if flipz:
            xi=xi[:,:,::-1]                                 
        if transpose:
            xi=np.transpose(xi,axes=(1,0,2))            
        (nx, ny, nz) = xi.shape;
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

    
