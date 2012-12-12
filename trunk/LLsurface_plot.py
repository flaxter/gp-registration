from numpy.random import randn, rand
from quaternions import transformPts

from gp import generate, getLogL_chol, shuffleIt, gp, gp_bootstrap

from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import cm

def generate2(X, Y):
    return X**2 + Y**2 

def LLsurface_plot(randPoints=True, n1=100, n2=100, generate=generate, sigma=0.001):

    # generate some scene data Xs, Ys, Zs
    Xs = 2*rand(n1) - 1
    Ys = 2*rand(n1) - 1
    Zs = generate(Xs, Ys)
        
    # now generate some test points X, Y, Z
    if randPoints == True: 
        X = 2*rand(n2) - 1
        Y = 2*rand(n2) - 1
    else:
        X, Y = np.meshgrid(np.linspace(-1, 1, sqrt(n2)), 
                           np.linspace(-1, 1, sqrt(n2)))
                           
    Z = generate(X, Y)
    
    ty_lst = np.linspace(-2,2,50)
    rotZ_lst = np.linspace(-20,20,50)
    
    LLmap = np.zeros((ty_lst.shape[0],rotZ_lst.shape[0]))
    
    K, L, alpha = gp_bootstrap(np.concatenate([[Xs],[Ys]]).T, Zs, sigma=sigma)
    
    for i, ty in enumerate(ty_lst):
        for j, rotZ in enumerate(rotZ_lst):
            #then rotate it in z and translate by ty
            XYZ = np.concatenate([[X.flatten()],[Y.flatten()],[Z.flatten()]]).T
            XYZ_transformed = np.array(transformPts(XYZ, [0.0,0.0,rotZ], [0.0,ty,0.0])[0])
            
            # now get the predictions for the X, Y using Xs, Ys, Zs
            mean, var = gp(np.concatenate([[Xs],[Ys]]).T, 
                           Zs, 
                           XYZ_transformed[:,:2],
                           K, L, alpha,
                           sigma=sigma)
            
            #now systematically get the logL for the data over a grid of (rotZ, ty) transformations
            LLmap[j, i] = getLogL_chol(mean, var, XYZ_transformed[:,2])
            print i, j, '\t', (ty,rotZ)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    plotX, plotY = np.meshgrid(ty_lst,rotZ_lst)
    ax.plot_wireframe(plotX, plotY, LLmap) #, linewidth=0, , alpha=0.5), cmap=cm.hot
    
    fig2 = plt.figure()    
    ax2 = Axes3D(fig2)
    ax2.plot_surface(plotX, plotY, LLmap, linewidth=0, cmap=cm.hot, rstride=2, cstride=2)

    fig3 = plt.figure()
    ax3 = Axes3D(fig3)
    Xp, Yp = np.meshgrid(np.linspace(-1, 1, 50), 
                         np.linspace(-1, 1, 50))   
    Zp = generate(Xp, Yp)
    ax3.plot_surface(Xp, Yp, Zp, alpha=0.75, linewidth=0, cmap=cm.hot, rstride=2, cstride=2)
    ax3.scatter(X,Y,Z, c='g', s=25)
    ax3.scatter(Xs,Ys,Zs, c='b', s=25)
    
    plt.show()

if __name__ == '__main__':
    LLsurface_plot(n1=500, n2=25)
