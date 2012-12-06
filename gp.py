from __future__ import division
from numpy import *
from numpy.linalg import norm, inv, cholesky, solve, det
from numpy.random import randn, rand
import time
            
def getK(X,Xp, k1=1, k2=1):
    k = zeros((X.shape[0],Xp.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Xp):  
            k[i,j] = k1*exp(-.5*k2*(norm(x-y)**2))
    return k
    
def gp_simple(X,y,pts,sigma=0.1):
    K = getK(X,X)
    Kstar = getK(X,pts)
    
    alpha = dot(Kstar.T, inv(K + (sigma**2)*eye(K.shape[0])))
    mean = dot(alpha,y)
    var = getK(pts,pts) - dot(alpha, Kstar) + (sigma**2)*eye(mean.shape[0])
    return mean, var
    
def gp_chol(X,y,pts,sigma=0.1):
    K = getK(X,X)
    Kstar = getK(X,pts)
    L = cholesky(K + (sigma**2)*eye(K.shape[0]))
    alpha = solve(L.T,solve(L,y))
    mean = dot(Kstar.T, alpha)
    v = solve(L,Kstar)
    var = getK(pts,pts) - dot(v.T,v) + (sigma**2)*eye(mean.shape[0])
    return mean, var
    
def shuffleIt(c):
    ''' this fixes the indices from the output of meshgrid so the covariance
        plots can be easier to interpret '''
    d = zeros_like(c)
    sq_c = sqrt(c.shape[0])
    for i in range(c.shape[0]):
        cc = c[i,:].reshape((sq_c,sq_c))
        ii = i//sq_c
        jj = i%sq_c
        d[ii*sq_c:ii*sq_c+sq_c,jj*sq_c:jj*sq_c+sq_c] = cc
    return d

def getLogL_chol(mean, cov, ptsY):
    ''' instead of calculating the determinant of the covariance matrix
        directly, which is numerical unstable, I am summing the log of the 
        diagonal of the cholesky factorization. I got this idea of chaper 2,
        algorithm 2.1 of the GP book '''
    D = mean.shape[0]
    L = cholesky(cov)

    deltMean = ptsY - mean
    LL = sum(log(diag(L)))
    LL += .5*dot(deltMean.T,solve(L.T,solve(L,deltMean)))
    LL += D/2.0*log(2*3.1415926535)
    return LL
    
def getLogL_simple(mean, cov, ptsY):
    ''' Don't use this one: the determinant of the cov 
          matrix is going to zero and blowing up the LL '''
    D = mean.shape[0]
    invCov = inv(cov)

    deltMean = ptsY - mean
    LL = .5*log(det(cov))
    LL += .5*dot(deltMean.T, dot(invCov, deltMean))
    LL += D/2.0*log(2*3.1415926535)
    return LL

def case1D():
    import pylab as pl
    import numpy as np
    def _f(x):
        """The function to predict."""
        return x * sin(x)
    
    sigma = 0.1
    X = np.linspace(0.1, 9.9, 5)
    X = np.atleast_2d(X).T

    # Observations and noise
    y = _f(X).ravel()
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise

    x = np.atleast_2d(np.linspace(0, 10, 300)).T

    #st = time.time()
    mean1, var1 = gp_simple(X, y, x, sigma)
    #print time.time() - st; st = time.time()
    #mean2, var2 = gp_chol(X, y, x, sigma)
    #print time.time() - st

    fig = pl.figure()
    pl.plot(x, _f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
    pl.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
    pl.plot(x, mean1, 'b-', label=u'Prediction')
    pl.fill(np.concatenate([x, x[::-1]]), \
            np.concatenate([mean1 - 1.9600 * diag(var1),
                           (mean1 + 1.9600 * diag(var1))[::-1]]), \
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    pl.xlabel('$x$')
    pl.ylabel('$f(x)$')
    pl.ylim(-10, 20)
    pl.legend(loc='upper left')

    pl.show()
    
def case2D(plotIt=True, randPoints=False):
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    from matplotlib import cm

    def generate(X, Y, phi):
        R = 1 - np.sqrt(X**2 + Y**2)
        return np.cos(2 * np.pi * X + phi) * R      
        
    # pick a whole bunch of random points (x,y) and then generate z + noise
    xs = 2*rand(1000) - 1
    ys = 2*rand(1000) - 1
    zs = generate(xs, ys, 0.0)
   
    # now pick some new test points  
    if randPoints == True: 
        X = 2*rand(100) - 1
        Y = 2*rand(100) - 1
    else:
        X, Y = np.meshgrid(np.linspace(-1, 1, 10), 
                           np.linspace(-1, 1, 10))
                           
    Z = generate(X, Y, 0.0)
          
    # now predict the z component from the xs,ys,zs
    st = time.time()
    mean, var = gp_chol(np.concatenate([[xs],[ys]]).T, zs, np.concatenate([[X.flatten()],[Y.flatten()]]).T, sigma=.001)
    print 'time chol:', time.time() - st; st = time.time()
    #mean2, var2 = gp_simple(np.concatenate([[xs],[ys]]).T, zs, np.concatenate([[X.flatten()],[Y.flatten()]]).T, sigma=.001)
    #print 'time simple:', time.time() - st
        
    

    if plotIt == True:    
        fig = plt.figure()
        plt.plot(linspace(-1,1,100), [getLogL_chol(mean, var, Z.flatten()+i) for i in linspace(-1,1,100)]) 
        plt.title('Varying $t_z$ from -1 to 1')
        plt.ylabel('Negative Log-Likelihood')
        plt.xlabel('$t_z$')
    
        fig = plt.figure()

        var_masked = var.copy()
        var_masked[var <= 0] = 1e-10
        plt.imshow(shuffleIt(log(var_masked)), interpolation='nearest') 
    
        fig = plt.figure()

        #ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        
        scene = ax.scatter(xs, ys, zs, c='r')
        
        #predicted = ax.plot_surface(X, Y, mean.reshape(X.shape), linewidth=0, cmap=cm.hot, alpha=0.5)
        predicted = ax.plot_wireframe(X, Y, mean.reshape(X.shape), color='g')

        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(xs, ys)
        Z = generate(X, Y, 0.0)

        wframe = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        
    plt.show()
    
    
    
if __name__ == '__main__':
    #case1D()
    case2D()

