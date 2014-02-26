from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor

#from mpl_toolkits.axes_grid1 import ImageGrid

def subplot_stack(x,Y,fig=None,ncols=None,nrows=None,title='',colors=['k','b'],hspace=0,wspace=0,use_yticks=False,use_xticks=True,enumerate_subplots=False,ytick_bins=None,xtick_bins=None):
    """ Stack a series of 1D plots
        
        support for calls to an existing fig is still experimental.  should 
        work in case where x is the same both times and Y has the same dimensions
        each time
    """
    x=np.asanyarray(x).squeeze()
    Y=np.asanyarray(Y).squeeze()
    
    if x.ndim>1:
        raise ValueError("x must have only one non-singleton dimension")
    
    if Y.ndim==1:
        Y=Y[:,np.newaxis]
    if Y.ndim>2:
        raise ValueError("Y must be 1D or 2D")
    if Y.shape[0]!=x.shape[0]:
        raise ValueError("x and Y shapes incompatible")
    
    if nrows and ncols:
        raise ValueError("specify either nrows or ncols, but not both")
    if ncols:
        nrows=int(ceil(Y.shape[1]/ncols))
    if nrows:
        ncols=int(ceil(Y.shape[1]/nrows))
    else: #default to ncols=1
        ncols=1
        nrows=Y.shape[1]
  
    if isinstance(colors,str):
        colors=[colors,]
        
    if np.iscomplexobj(Y):
        is_complex=True
        if len(colors)==1: #if only 1 color specified, use it for the imaginary axis too
            colors=[colors[0],]*2
    else:
        is_complex=False
            
    middle_col=int(floor(ncols/2))
    
    if not fig:
        fig=plt.figure()
        axes_list=[]
        use_existing_fig=False
    else:
        use_existing_fig=True
        axes_list=fig.axes
    
    cnt=0;
    for r in range(nrows):
        for c in range(ncols):
            #axes_list.append(plt.subplot2grid((nrows,ncols),(r,c),colspan=1,rowspan=1,sharex=axes_list[0]))
            if not use_existing_fig:
                axes_list.append(plt.subplot2grid((nrows,ncols),(r,c),colspan=1,rowspan=1))
#            else:
#                if cnt>len(axes_list):
                    #raise ValueError(                
              
            #get current axis  
            ax=axes_list[cnt]
             
            if use_existing_fig:
                ax.hold(True)
                
            if is_complex:
                ax.plot(x,Y[:,cnt].real,colors[0],x,Y[:,cnt].imag,colors[1])
            else:
                ax.plot(x,Y[:,cnt],colors[0])
                
            if enumerate_subplots: #number the subplots
                ax.set_ylabel('{}   '.format(cnt),rotation='horizontal')
                #ax.set_ylabel('Y[:,{}]'.format(cnt),rotation='vertical')
            cnt+=1
               
            if use_xticks:
                if xtick_bins:
                    ax.locator_params(axis='x',nbins=xtick_bins)
                if ncols>1 and wspace<0.25: # help avoid y-ticks overlap by hiding the bottom tick
                    ax.set_xticks(ax.get_xticks()[1:])  
            if r<(nrows-1) or (not use_xticks):   #remove xticks from all but bottommost
                    plt.setp(ax.get_xticklabels(), visible=False) 
                
            if use_yticks:
                if ytick_bins:
                    ax.locator_params(axis='y',nbins=ytick_bins)
                if nrows>1 and hspace<0.25: # help avoid y-ticks overlap by hiding the bottom tick
                    ax.set_yticks(ax.get_yticks()[1:])  
                if c>0: #remove y-ticks from all but right-most axis
                    plt.setp(ax.get_yticklabels(), visible=False)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
    
            if title and r==0 and c==middle_col:
                ax.set_title(title)

    fig.subplots_adjust(hspace=hspace,wspace=wspace)        
  
    fig.show()
    return fig