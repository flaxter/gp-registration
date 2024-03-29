from numpy.random import randn, rand
from quaternions import transformPtsQ, transformPts
from numpy.linalg import norm, inv, cholesky, solve, det
from gp import generate, getLogL_chol, shuffleIt, gp, gp_bootstrap, getSceneAndNew, gradientDescent

from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
from matplotlib import cm

import vectors
from vectors import M, Q

from numpy import *

from quaternions import rotQ, getQ

def generate2(X, Y, phi=0):
    return X**2 + Y**2 
    
def generate3(X, Y, phi=0):
    return abs(X) + abs(Y)

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
    
    ty_lst = np.linspace(-1,1,50)
    rotZ_lst = np.linspace(-90,90,50)
    
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
    #ax.plot_wireframe(plotX, plotY, LLmap) #, linewidth=0, , alpha=0.5), cmap=cm.hot
    ax.plot_surface(plotX, plotY, LLmap, linewidth=0.5, cmap=cm.Blues, rstride=1, cstride=1, alpha=0.3)
    
    cset = ax.contour(plotX, plotY, LLmap, zdir='z', offset=0, cmap=cm.coolwarm)
    #cset = ax.contour(plotX, plotY, LLmap, zdir='x', offset=-1, cmap=cm.coolwarm)
    #cset = ax.contour(plotX, plotY, LLmap, zdir='y', offset=90, cmap=cm.coolwarm)

    ax.set_xlabel('$t_y$')
    ax.set_xlim(-1, 1)
    ax.set_ylabel('Rotation around $z$-axis (degrees)')
    ax.set_ylim(-90, 90)
    ax.set_zlabel('Negative Log-Likelihood')
    ax.set_zlim(0, 5e6)
    
    fig2 = plt.figure()    
    ax2 = Axes3D(fig2)
    ax2.plot_surface(plotX, plotY, LLmap, linewidth=0.5, cmap=cm.hot, rstride=1, cstride=1, alpha=0.3)
    
    cset = ax2.contour(plotX, plotY, LLmap, zdir='z', offset=0, cmap=cm.coolwarm)
    #cset = ax2.contour(plotX, plotY, LLmap, zdir='x', offset=-1, cmap=cm.coolwarm)
    #cset = ax2.contour(plotX, plotY, LLmap, zdir='y', offset=90, cmap=cm.coolwarm)

    ax2.set_xlabel('$t_y$')
    ax2.set_xlim(-1, 1)
    ax2.set_ylabel('Rotation around z-axis (degrees)')
    ax2.set_ylim(-90, 90)
    ax2.set_zlabel('Negative Log-Likelihood')
    ax2.set_zlim(0, 5e6)

    '''fig3 = plt.figure()
    ax3 = Axes3D(fig3)
    Xp, Yp = np.meshgrid(np.linspace(-1, 1, 50), 
                         np.linspace(-1, 1, 50))   
    Zp = generate(Xp, Yp)
    ax3.plot_surface(Xp, Yp, Zp, alpha=0.75, linewidth=0, cmap=cm.hot, rstride=1, cstride=1)
    ax3.scatter(X,Y,Z, c='g', s=25)
    ax3.scatter(Xs,Ys,Zs, c='b', s=25)'''
    
    plt.show()
    
def animPoints():
    sigma = 0.1

    T_vector = [.25,-.2,.3]
    qReal = Q.rotate('Z', vectors.radians(-15)) * Q.rotate('Y', vectors.radians(-15))
    u, v, w, s = qReal
    qReal = array([s, u, v, w])
    print qReal
    print 'Translation', T_vector
    X,y,pts,z = getSceneAndNew(T_vector=T_vector, 
                               qReal=qReal, n1=100, n2=100, 
                               sigma=sigma, generate=generate)

    q, t, LL, Ts = gradientDescent(X,y,pts,z,sigma=sigma, iterations=150, beta=0.01, returnTraj=True, numerical=False)
    print 'Real Translation', around(T_vector, decimals=2)
    print 'Real Rotation'
    print around(rotQ(qReal), decimals=2)
    print 'Calculated t', around(-dot(inv(rotQ(q)),t), decimals=2)
    print 'Calculated R'
    print around(linalg.inv(rotQ(q)), decimals=2)
    
    plt.ion()
    fig = plt.figure()
    ax = Axes3D(fig)    
    ax.scatter(X[:,0], X[:,1], y, c='r')
    
    animPts = None
    while True:
        for T in Ts: 
            old = animPts
            
            pts3d = concatenate((pts, atleast_2d(z).T), axis=1)
            ptsNew = transformPtsQ(pts3d, T[3], array([T[0], T[1], T[2]]))
            
            animPts = ax.scatter(ptsNew[:,0], 
                                 ptsNew[:,1], 
                                 ptsNew[:,2])
                                 
            if old is not None:
                ax.collections.remove(old)
                                 
            plt.draw()
    plt.show()
    
def getMSE(pts1, pts2):
    err = 0
    for pt1, pt2 in zip(pts1, pts2):
        err += (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2
    return err/len(pts1)
    
def getFrobErr(R,t,R2,t2):
    return sqrt(sum((R-R2)**2)) + sqrt(sum((t-t2)**2))

from pylab import * 
from pickle import load   
def runTests():
    matplotlib.rcParams.update({'font.size': 16})
    sigma=0.1
    outF = open('convergenceDumpGP','r')
    (tParams,resMSE,resFrob) = load(outF)
    
    print resMSE.shape
    
    mseLst = []
    
    N = 100
    
    for i, T in enumerate(tParams[:N]):
        qReal = getQ(T[:3])
        T_vector = T[3:]
        X,y,pts,z = getSceneAndNew(T_vector=T_vector, 
                                   qReal=qReal, n1=100, n2=20, 
                                   sigma=sigma, generate=generate)
        q, t, LL, Ts = gradientDescent(X,y,pts,z,sigma=sigma, iterations=50, beta=0.01, returnTraj=True)
        
        XYZ2 = transformPtsQ(concatenate((pts, atleast_2d(z).T), axis=1), q, t)
        XYZ = concatenate((X, atleast_2d(y).T), axis=1)
        
        mse = getMSE(XYZ, XYZ2)
        mseLst.append(mse)
        #print mse, resMSE[0,i]
        
    mseLst = array(mseLst)
    
    print mseLst.shape, resMSE[:,:].shape
        
    mse = vstack((resMSE[:,:N], mseLst.reshape((1,len(mseLst)))))
    
    styles = ['r','g','b','k--.']
    markers = ['x','o','+','^']
    
    figure()
    
    '''hist(mse[1:3,:].T)
    legend(['ICP','GP'])'''
    #title('Convergence Results')    
    color = ['b','g','r']
    legendLbl = ['','ICP','GP']
    
    for i in range(3):
        if i == 1 or i == 2:
            subplot(1,2,i)
            hist(mse[i,:], color=color[i], bins=arange(0,5.0,0.35))
            ylim(0,100.0)
            xlabel("Mean Squared Error (MSE)")
            print mse[i,:]
            if i==1: ylabel("Frequency (out of 100)")
            legend([legendLbl[i]], loc='upper right')
            #X = sorted(mse[i,:])
            #plot(X,[v/100. for v in arange(N)], styles[i], lw=7) # marker=markers[i], ms=5
    
    #ylabel("p(MSE < x)")
    
    
           
    show()  
    
def avg(lst): return 1.0*sum(lst)/len(lst)

def stddev(lst): 
    a = avg(lst)
    return sqrt(sum([(x-a)**2 for x in lst]))   
    
def runScalingTests():
    matplotlib.rcParams.update({'font.size': 16})
    outF = open('scalingDumpGP','r')
    (tParams,rr, N, t1, dev) = load(outF)
    
    sigma = 0.1
    
    t2 = []
    err = []
    
    stopN = 20
    
    for nPts in N[:stopN]:
        tt2 = []
        for i, T in enumerate(tParams[:3]):
            
            qReal = getQ(T[:3])
            T_vector = T[3:]
            X,y,pts,z = getSceneAndNew(T_vector=T_vector, 
                                       qReal=qReal, n1=nPts*.25, n2=20, 
                                       sigma=sigma, generate=generate)
            st = time.time()
            q, t, LL, Ts = gradientDescent(X,y,pts,z,sigma=sigma, iterations=15, beta=0.01, returnTraj=True)  
            tt = time.time() - st
            
            print nPts, tt
            tt2.append(tt)
        t2.append(avg(tt2))
        err.append(stddev(tt2))
            
    print len(t), len(t2), len(N)
    
    
    errorbar(N[:stopN], t1[:stopN], yerr=dev[:stopN], linewidth=6, elinewidth=1)
    errorbar(N[:stopN], t2[:stopN], yerr=err[:stopN], linewidth=6, elinewidth=1)
    #title('Convergence')    
    xlabel('Number of Points')
    ylabel('Time until Convergence (s)')
    legend(['icp', 'gp'], loc='best') #, 'emicp-openmp', 'emicp-gpu', 'softassign'])
    
    show()   
        

if __name__ == '__main__':
    #LLsurface_plot(n1=500, n2=25)
    #animPoints()
    
    runTests()
    #runScalingTests()
