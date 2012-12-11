from __future__ import division
from numpy import *
from numpy.linalg import norm, inv, cholesky, solve, det
from numpy.random import randn, rand
import time
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from quaternions import dRotate
            
def getK_slow(X, Xp, k1=1, k2=1):
    k = zeros((X.shape[0],Xp.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Xp):  
            k[i,j] = k1*exp(-.5*k2*(norm(x-y)**2))
    return k
    
def getK(X, Xp, k1=1, k2=1):
    return k1*exp(-.5*k2*cdist(X, Xp)**2)
    
def getDK(X, Xp, Xnew, q, k1=1, k2=1):
    # reeeallly subtle broadcasting here
    # in matlab I would do a diag(X)*K*diag(Y) sort of thing
    K = k1*exp(-.5*k2*cdist(X, Xp)**2)
    dk_x = (-X[:,0]*K.T).T + Xp[:,0]*K
    dk_y = (-X[:,1]*K.T).T + Xp[:,1]*K
    
    ds, du, dv, dw = dRotate(q, Xnew[:,0], Xnew[:,1], Xnew[:,2])
    
    ds, du, dv, dw = array(ds).T, array(du).T, array(dv).T, array(dw).T
    
    dk_s = ((-X[:,0]*K.T).T + Xp[:,0]*K)*ds[:,0] + ((-X[:,1]*K.T).T + Xp[:,1]*K)*ds[:,1]
    dk_u = ((-X[:,0]*K.T).T + Xp[:,0]*K)*du[:,0] + ((-X[:,1]*K.T).T + Xp[:,1]*K)*du[:,1]
    dk_v = ((-X[:,0]*K.T).T + Xp[:,0]*K)*dv[:,0] + ((-X[:,1]*K.T).T + Xp[:,1]*K)*dv[:,1]
    dk_w = ((-X[:,0]*K.T).T + Xp[:,0]*K)*dw[:,0] + ((-X[:,1]*K.T).T + Xp[:,1]*K)*dw[:,1]
  
    return dk_x, dk_y, dk_s, dk_u, dk_v, dk_w # note that dk_z is zero!
    
def gp_simple(X,y,pts,sigma=0.1):
    ''' don't use this one. it is for simple testing purposes on small data
    '''
    K = getK(X,X)
    Kstar = getK(X,pts)
    
    alpha = dot(Kstar.T, inv(K + (sigma**2)*eye(K.shape[0])))
    mean = dot(alpha,y)
    var = getK(pts,pts) - dot(alpha, Kstar) + (sigma**2)*eye(mean.shape[0])
    return mean, var
    
def gp_chol(X,y,pts,sigma=0.1):

    K = getK(X,X)
    Kstar = getK(X,pts)   
    Kstarstar = getK(pts,pts)

    L = cholesky(K + (sigma**2)*eye(K.shape[0]))
    alpha = solve(L.T,solve(L,y))
 
    mean = dot(Kstar.T, alpha)
    
    v = solve(L,Kstar)   
    var = Kstarstar - dot(v.T,v) + (sigma**2)*eye(mean.shape[0])
 
    return mean, var
    
def gp_bootstrap(X,y,pts,sigma=0.1):
    K = getK(X,X)
    Kstarstar = getK(pts,pts)
    L = cholesky(K + (sigma**2)*eye(K.shape[0]))
    alpha = solve(L.T,solve(L,y))

    return K, Kstarstar, L, alpha

def gp(X, y, pts, K, Kstarstar, L, alpha, sigma=0.1):
    Kstar = getK(X,pts)    
    mean = dot(Kstar.T, alpha)
    
    v = solve(L,Kstar)   
    var = Kstarstar - dot(v.T,v) + (sigma**2)*eye(mean.shape[0])
 
    return mean, var
    
def gradientDescent(X,y,pts,z,sigma=0.1,verbose=2,iterations=250):
    print X.shape
    print y.shape
    print pts.shape
    print z.shape
#    print X
#    print y
#    print pts
#    print z
    # these are constants in the calculations
    K, Kstarstar, L, alpha = gp_bootstrap(X,y,pts,sigma) #.4 seconds
    
    # K is K(X,X)
    # Kstarstar is K(X',X')
    # L is the lower triangular (K + sigma^2 I) matrix
    # alpha is (K + sigma^2 I)^-1 * y    
    
    XYZ = concatenate((pts, atleast_2d(z).T), axis=1)
    
    t_x, t_y, t_z = 0.0,0.0,0.0
    q = array([1.0,0.0,0.0,0.0])
    
    totalT = array([0.0,0.0,0.0])
    totalQ = array([0.0,0.0,0.0,0.0])
    if verbose > 1:
        print 'translation offset\t negative log likelihood (should be minimized)'
    for i in range(iterations):
        # we need to calculate the derivatives and move the (pts,z) around rigidly

        ######################
        # first term  
        ######################        
        
        Kstar = getK(X,pts)
        v = solve(L,Kstar) # L^-1 * Kstar
        
        mean = dot(Kstar.T, alpha)
        
        # v^T v = (L^-1 Kstar)^T (L^-1 Kstar) = kstar^T (K + sigma^2 I)^-1 kstar
        var = Kstarstar - dot(v.T,v) + (sigma**2)*eye(Kstarstar.shape[0])
        Lvar = cholesky(var)
        
        # tensor split into two matrices
        # the following calculations will come in pairs now
        dKstar_x, dKstar_y, dKstar_s, dKstar_u, dKstar_v, dKstar_w = getDK(X, pts, XYZ, q) 
        
        # calculating derivative of the first term:
        v_dKstar_x = solve(L, dKstar_x) # L^-1 dKstar1 (for x)
        v_dKstar_y = solve(L, dKstar_y) # L^-1 dKstar2 (for y)
           
        # -Kstar^T (K + sigma^2 I)^-1 dKstar1 - DKstar^T (K + sigma^2 I)^-1 Kstar1 
        dCov_x = -dot(v.T, v_dKstar_x) - dot(v_dKstar_x.T, v) # (for x)
        dCov_y = -dot(v.T, v_dKstar_y) - dot(v_dKstar_y.T, v) # (for x)
        
        # (var)^-1 (-2 * Kstar^T (K + sigma^2 I)^-1 dKstar1)
        Dx = solve(Lvar.T, solve(Lvar, dCov_x)) 
        Dy = solve(Lvar.T, solve(Lvar, dCov_y))
        
        # as per Seth's equations, we take the trace of these bad boys
        dx = trace(Dx) 
        dy = trace(Dy)
        
        ######################
        # now the second term  
        ######################
          
        # errors in z
        delt = z - mean
        
        # a lot packed in here...
        # -2 (z - mean)^T (var)^-1 dKstar^T (K + sigma^2 I)^-1 * y   
        dx -= -2*dot(delt.T,solve(Lvar.T, solve(Lvar, dot(dKstar_x.T, alpha))))
        dy -= -2*dot(delt.T,solve(Lvar.T, solve(Lvar, dot(dKstar_y.T, alpha))))
          
        # even more here...
        # - (z - mean)^T (var)^-1 (-2 * Kstar^T (K + sigma^2 I)^-1 dKstar1) (var)^-1 (z - mean)
        dx -= dot(dot(delt.T, Dx), solve(Lvar.T, solve(Lvar, delt)))
        dy -= dot(dot(delt.T, Dy), solve(Lvar.T, solve(Lvar, delt)))
        
        # z is super easy...
        Dz = ones((mean.shape[0],1))
        v1 = solve(Lvar, delt) # L^-1 (z - mean)
        v2 = solve(Lvar, Dz)   # L^-1 (1's)
        
        # (z - mean)^T (cov)^-1 1's 
        # (the one's vector just sums it up)
        dz = 2*dot(v1.T, v2)[0]
        
        N = mean.shape[0]
        rho = .05
        stepX, stepY, stepZ = rho*dx/(N*N), rho*dy/(N*N), rho*dz/(N*N)
#        stepX, stepY, stepZ = .00005*dx/(N*N), .00005*dy/(N*N), .00005*dz/(N*N)
        
        z -= stepZ
        
        ### NOTE THAT i HAVE TURNED OFF UPDATES TO X UNTIL BUG IS FIXED
        pts -= array([stepX, stepY])
 
        totalT += array([stepX, stepY, stepZ])
        
        #print stepX, stepY, stepZ
        
        mean, var = gp(X, y, pts, K, Kstarstar, L, alpha, sigma=0.1)
        if verbose > 1:
            print totalT, '\t', getLogL_chol(mean, var, z)

    likelihood = getLogL_chol(mean, var, z)
    if verbose > 0:
        print totalT, '\t', likelihood

    return(totalT, likelihood)

        
    
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

def generate(X,Y,phi):
    return X

def applyTransform(D,T):
    import numpy as np
#    print D
#    print T
#    print (D[0]+T[0],D[1]+T[1],D[2]+T[2])
    return (D[0]+T[0],D[1]+T[1],D[2]+T[2])

def getAvgClosestDistance(X):
    import numpy as np

    XX=array(X).T
    Y=pdist(XX,'euclidean')
    for i in range(len(Y)):
        if Y[i]==0.:
            Y[i]=999999

    Y=squareform(Y)
    Y=Y+np.diag(zeros(Y.shape[0])+999999)
    return mean(Y.min(0))

def generateTransforms():
    import numpy as np

    Tx=[]
    for i in range(5):
	Tx.append((rand()*30,0,0))

    Ty=[]
    for i in range(5):
	Ty.append((0,rand()*30,0))

    Txy=[]
    for i in range(5):
	Txy.append((rand()*30,rand()*30,0))

    Tz=[]
    for i in range(5):
	Tz.append((0,0,rand()*30))

    Txyz=[]
    for i in range(5):
	Txyz.append((rand()*30,rand()*30,rand()*30))

    T=[]
    T.extend(Ty)
    T.extend(Tx)
    T.extend(Txy)
    T.extend(Tz)
    T.extend(Txyz)
#    print np.dim(T)
    return T

def generateSmooth(X,Y,phi):
    import numpy as np
#    X, Y = np.meshgrid(np.linspace(-1,1,20),
#			np.linspace(-1,1,20))
#    phi = 0.1;
    R = 1 - np.sqrt(X**2 + Y**2)
#    print X*Y
    Z = np.cos(2 * np.pi * X/3 * Y/3 + phi) * R
    return Z

def generateAbs(X,Y,phi):
    import numpy as np
#    X, Y = np.meshgrid(np.linspace(-1,1,20),
#			np.linspace(-1,1,20))
    Z=abs(X-mean(X))+abs(Y-mean(Y))+phi
    return Z

def generatePlanes(numSurfs,n,p):
    import numpy as np
    X=np.zeros(0)
    Y=np.zeros(0)


    R = 1 - np.sqrt(X**2 + Y**2)
#    print X*Y
    Z = np.cos(2 * np.pi * X/3 * Y/3 + phi) * R
    return Z

def generateSmooth(X,Y,phi):
    import numpy as np
#    X, Y = np.meshgrid(np.linspace(-1,1,20),
#			np.linspace(-1,1,20))
#    phi = 0.1;
    R = 1 - np.sqrt(X**2 + Y**2)
#    print X*Y
    Z = np.cos(2 * np.pi * X/3 * Y/3 + phi) * R
    return Z

def generateAbs(X,Y,phi):
    import numpy as np
#    X, Y = np.meshgrid(np.linspace(-1,1,20),
#			np.linspace(-1,1,20))
    Z=abs(X-mean(X))+abs(Y-mean(Y))+phi
    return Z

def generatePlanes(numSurfs,n,p):
    import numpy as np
    X=np.zeros(0)
    Y=np.zeros(0)
    Z=np.zeros(0)
    for i in range(numSurfs):
	x,y=rand(2)*10
	w,h=rand(2)*2
	phi=rand(1)*30
	X2,Y2=np.meshgrid(np.linspace(x,x+w,np.sqrt(n)),np.linspace(y,y+h,np.sqrt(n)))
	if rand(1) > p:
	    Z2 = generateAbs(X2,Y2,phi)
	else:
	    Z2 = generateSmooth(X2,Y2,phi)

#	Z2 = generateSmooth(X2,Y2,phi)
#	print X2.flatten()
#	print X.flatten()
	X=concatenate((X.flatten(),X2.flatten()))
	Y=concatenate((Y.flatten(),Y2.flatten()))
	Z=concatenate((Z.flatten(),Z2.flatten()))
#    X=concat(X)
#    Y=concat(Y)
#    Z=concat(Z)
    return (X,Y,Z)

def case2D2(plotIt=True, randPoints=False, T_vector = [0,1.5,-.5]):
    from mpl_toolkits.mplot3d import axes3d, Axes3D
#    import matplotlib.pyplot as plt
    import numpy as np
    import time
    from matplotlib import cm
        
    # pick a whole bunch of random points (x,y) and then generate z + noise
    xs = 2*rand(n1) - 1
    ys = 2*rand(n1) - 1
    zs = generate(xs, ys, 0.0)
   
    # now pick some new test points  
    if randPoints == True: 
        X = 2*rand(n2) - 1
        Y = 2*rand(n2) - 1
    else:
        X, Y = np.meshgrid(np.linspace(-1, 1, n2), 
                           np.linspace(-1, 1, n2))

    Z = generate(X, Y, 0.0)

    # now predict the z component from the xs,ys,zs
    st = time.time()

    mean, var = gp_chol(np.concatenate([[xs],[ys]]).T, zs, np.concatenate([[X.flatten()],[Y.flatten()]]).T, sigma=.001)
    #print 'time chol:', time.time() - st; st = time.time()
    #mean2, var2 = gp_simple(np.concatenate([[xs],[ys]]).T, zs, np.concatenate([[X.flatten()],[Y.flatten()]]).T, sigma=.001)
    #print 'time simple:', time.time() - st

    return (np.concatenate([[xs],[ys]]).T, 
                    zs, 
                    np.concatenate([[X.flatten()+T_vector[0]],[Y.flatten()+T_vector[1]]]).T, 
                    Z.flatten()+T_vector[2],
                    .001)

def case2D(plotIt=True, randPoints=False, T_vector = [0,1.5,-.5], generate=generate, n1 = 1000, n2=100):
    from mpl_toolkits.mplot3d import axes3d, Axes3D
#    import matplotlib.pyplot as plt
    import numpy as np
    import time
    from matplotlib import cm
        
    # pick a whole bunch of random points (x,y) and then generate z + noise
    xs = 2*rand(n1) - 1
    ys = 2*rand(n1) - 1
    zs = generate(xs, ys, 0.0)
   
    # now pick some new test points  
    if randPoints == True: 
        X = 2*rand(n2) - 1
        Y = 2*rand(n2) - 1
    else:
        X, Y = np.meshgrid(np.linspace(-1, 1, n2), 
                           np.linspace(-1, 1, n2))
                           
    Z = generate(X, Y, 0.0)
          
    # now predict the z component from the xs,ys,zs
    st = time.time()
    
    mean, var = gp_chol(np.concatenate([[xs],[ys]]).T, zs, np.concatenate([[X.flatten()],[Y.flatten()]]).T, sigma=.001)
    #print 'time chol:', time.time() - st; st = time.time()
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
    
    return (np.concatenate([[xs],[ys]]).T, 
                    zs, 
                    np.concatenate([[X.flatten()+T_vector[0]],[Y.flatten()+T_vector[1]]]).T, 
                    Z.flatten()+T_vector[2],
                    .001)
    
#if __name__ == '__main__':
if True:
#    case1D()
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    import pylab as pl
    import numpy as np
    X,Y,Z=generatePlanes(3,36,0.5)
    XAbs,YAbs,ZAbs=generatePlanes(1,40,0)
    XCos,YCos,ZCos=generatePlanes(1,40,1)
    XSurfs,YSurfs,ZSurfs=generatePlanes(1,40,0)
    
    Ds=[]
    Ds.append((X,Y,Z))
    Ds.append((XAbs,YAbs,ZAbs))
    Ds.append((XCos,YCos,ZCos))
    Ds.append((XSurfs,YSurfs,ZSurfs))

    
    Ts=generateTransforms()


    for D in Ds:
	delta=getAvgClosestDistance(D)
        fig=pl.figure()
        ax=Axes3D(fig)
	for T in Ts:
            print 'Transformation is:'
            print T
            TD = applyTransform(D,T)
#	    ax.clear()
#            ax.plot(D[0],D[1],D[2],'b.')
#            ax.plot(TD[0],TD[1],TD[2],'r.')
#	    pl.savefig('gp-dataset.png')
#            break
##           Pass in D,TD to the GP function, return THAT
#            x,y,pts,z,sigma = case2D2(randPoints=True, plotIt=False,T_vector=T)
            sigma=delta/10
#            import code; code.interact(local=locals())
            (t,l) = gradientDescent(array(D[0:2]).T,D[2],array(TD[0:2]).T,TD[2],sigma,verbose=1)
##           Compare to T, evaluate accuracy
            print T
            print t
            print t-T,sqrt((t[0]-T[0])**2+(t[1]-T[1])**2+(t[2]-T[2])**2)
            print
#
##        pl.show()

    import code; code.interact(local=locals())


#    print X,Y,Z
#    fig = pl.figure()
#    ax = Axes3D(fig)
#    ax.plot(X,Y,Z,'b.')
#    pl.show()
#    T_vector = [.1,-.1,-.2]
#    print 'Translation', T_vector
#    X,y,pts,z,sigma = case2D(randPoints=True, plotIt=False, T_vector=T_vector, n1=100, n2=100)
#    gradientDescent(X,y,pts,z,sigma)

