from __future__ import division
from numpy import *
from numpy.linalg import norm, inv, cholesky, solve, det
from numpy.random import randn, rand
import time
from scipy.spatial.distance import cdist
from quaternions import dRotate, normalizeQ, transformPtsQ, rotQ
import vectors
from vectors import M, Q
            
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
    
    dk_s = ((-X[:,0]*K.T).T + Xp[:,0]*K)*ds[:,0] + ((-X[:,1]*K.T).T + Xp[:,1]*K)*ds[:,1]
    dk_u = ((-X[:,0]*K.T).T + Xp[:,0]*K)*du[:,0] + ((-X[:,1]*K.T).T + Xp[:,1]*K)*du[:,1]
    dk_v = ((-X[:,0]*K.T).T + Xp[:,0]*K)*dv[:,0] + ((-X[:,1]*K.T).T + Xp[:,1]*K)*dv[:,1]
    dk_w = ((-X[:,0]*K.T).T + Xp[:,0]*K)*dw[:,0] + ((-X[:,1]*K.T).T + Xp[:,1]*K)*dw[:,1]
  
    return dk_x, dk_y, dk_s, dk_u, dk_v, dk_w # note that dk_z is zero!
    
def getDKstarstar_slow(Xp, Xnew, q, k1=1, k2=1):
    # reeeallly subtle broadcasting here
    # in matlab I would do a diag(X)*K*diag(Y) sort of thing
    K = k1*exp(-.5*k2*cdist(Xp, Xp)**2)
    
    ds, du, dv, dw = dRotate(q, Xnew[:,0], Xnew[:,1], Xnew[:,2])
    
    dk_s, dk_u, dk_v, dk_w = (zeros((Xp.shape[0], Xp.shape[0])), 
                              zeros((Xp.shape[0], Xp.shape[0])),
                              zeros((Xp.shape[0], Xp.shape[0])),
                              zeros((Xp.shape[0], Xp.shape[0])))
                             
    for i in range(Xp.shape[0]):
        for j in range(Xp.shape[0]):
            dk_s[i,j] = K[i,j]*dot((Xp[i,:] - Xp[j,:]).T,ds[i,:2] - ds[j,:2])
            dk_u[i,j] = K[i,j]*dot((Xp[i,:] - Xp[j,:]).T,du[i,:2] - du[j,:2])
            dk_v[i,j] = K[i,j]*dot((Xp[i,:] - Xp[j,:]).T,dv[i,:2] - dv[j,:2])
            dk_w[i,j] = K[i,j]*dot((Xp[i,:] - Xp[j,:]).T,dw[i,:2] - dw[j,:2])
  
    # note that dk_x, dk_y, and dk_z are zero!
    return dk_s, dk_u, dk_v, dk_w 
    
def getDKstarstar(Xp, Xnew, q, k1=1, k2=1):
    # reeeallly subtle broadcasting here
    # in matlab I would do a diag(X)*K*diag(Y) sort of thing
    K = k1*exp(-.5*k2*cdist(Xp, Xp, 'sqeuclidean'))
    
    ds, du, dv, dw = dRotate(q, Xnew[:,0], Xnew[:,1], Xnew[:,2])
          
    ds_diff_x = cdist(ds, ds, lambda u, v: u[0]-v[0])
    ds_diff_y = cdist(ds, ds, lambda u, v: u[1]-v[1])
    
    du_diff_x = cdist(du, du, lambda u, v: u[0]-v[0])
    du_diff_y = cdist(du, du, lambda u, v: u[1]-v[1])
    
    dv_diff_x = cdist(dv, dv, lambda u, v: u[0]-v[0])
    dv_diff_y = cdist(dv, dv, lambda u, v: u[1]-v[1])
    
    dw_diff_x = cdist(dw, dw, lambda u, v: u[0]-v[0])
    dw_diff_y = cdist(dw, dw, lambda u, v: u[1]-v[1])

    Xp_diff_x = cdist(Xp, Xp, lambda u, v: u[0]-v[0])
    Xp_diff_y = cdist(Xp, Xp, lambda u, v: u[1]-v[1])
    
    dk_s = K*(ds_diff_x*Xp_diff_x + ds_diff_y*Xp_diff_y)    
    dk_u = K*(du_diff_x*Xp_diff_x + du_diff_y*Xp_diff_y)
    dk_v = K*(dv_diff_x*Xp_diff_x + dv_diff_y*Xp_diff_y)
    dk_w = K*(dw_diff_x*Xp_diff_x + dw_diff_y*Xp_diff_y) 

    return dk_s, dk_u, dk_v, dk_w 
    
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
    
def gp_bootstrap(X,y,sigma=0.1):
    K = getK(X,X)
    L = cholesky(K + (sigma**2)*eye(K.shape[0]))
    alpha = solve(L.T,solve(L,y))

    return K, L, alpha

def gp(X, y, pts, K, L, alpha, sigma=0.1):
    Kstar = getK(X,pts) 
    Kstarstar = getK(pts,pts)   
    mean = dot(Kstar.T, alpha)
    
    v = solve(L,Kstar)   
    var = Kstarstar - dot(v.T,v) + (sigma**2)*eye(mean.shape[0])
 
    return mean, var
    
def gradientDescent(X,y,pts,z,sigma=0.1,verbose=2,iterations=35,beta=0.0005, returnTraj=False, numerical=True):
    print X.shape
    print y.shape
    print pts.shape
    print z.shape

    print type(X)
    print type(y)
    print type(pts)
    print type(z)

    print 'Hello'
    # these are constants in the calculations
    K, L, alpha = gp_bootstrap(X,y,sigma) #.4 seconds
    
    # K is K(X,X)
    # Kstarstar is K(X',X')
    # L is the lower triangular (K + sigma^2 I) matrix
    # alpha is (K + sigma^2 I)^-1 * y    
    
    Xnew = concatenate((pts, atleast_2d(z).T), axis=1)
    Ts=[]

    #tx, ty, tz = concatenate([X,y],1).mean(axis=0)-concatenate([pts,z],1).mean(axis=0)
    tx,ty,tz=0.0,0.0,0.0
    q = array([1.0,0.0,0.0,0.0])

    Ts.append((tx,ty,tz,q))

    LL_last = 99e9

    plotSteps=False
    if plotSteps:
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        import matplotlib.pyplot as pl
        from pylab import ion
        ion()
        fig=pl.figure()
        ax=Axes3D(fig)
        ax.plot(X[:,0],X[:,1],y,'g.')
        ax.plot(Xnew[:,0],Xnew[:,1],Xnew[:,2],'b.')
        pl.show()

    if verbose > 1:
        print 'translation offset\t negative log likelihood (should be minimized)'
    for i in range(iterations):
        # we need to calculate the derivatives and move the (pts,z) around rigidly

        ######################
        # first term  
        ######################   
        
        if not numerical:     
        
            Kstar = getK(X,pts)
            Kstarstar = getK(pts,pts)
            v = solve(L,Kstar) # L^-1 * Kstar
            
            mean = dot(Kstar.T, alpha)
            
            # v^T v = (L^-1 Kstar)^T (L^-1 Kstar) = kstar^T (K + sigma^2 I)^-1 kstar
            var = Kstarstar - dot(v.T,v) + (sigma**2)*eye(Kstarstar.shape[0])
            Lvar = cholesky(var)
            
            # tensor split into 6 matrices
            # the following calculations will come in groups of 6 now
            dKstar_x, dKstar_y, dKstar_s, dKstar_u, dKstar_v, dKstar_w = getDK(X, pts, Xnew, q) 
            
            # calculating derivative of the first term:
            v_dKstar_x = solve(L, dKstar_x) # L^-1 dKstar1 (for x)
            v_dKstar_y = solve(L, dKstar_y) # L^-1 dKstar2 (for y)
            v_dKstar_s = solve(L, dKstar_s)
            v_dKstar_u = solve(L, dKstar_u)
            v_dKstar_v = solve(L, dKstar_v)
            v_dKstar_w = solve(L, dKstar_w)
            
            dKstarstar_ds, dKstarstar_du, dKstarstar_dv, dKstarstar_dw = getDKstarstar(pts, Xnew, q)
               
            # -Kstar^T (K + sigma^2 I)^-1 dKstar1 - DKstar^T (K + sigma^2 I)^-1 Kstar1 
            dCov_x = -dot(v.T, v_dKstar_x) - dot(v_dKstar_x.T, v) # (for x)
            dCov_y = -dot(v.T, v_dKstar_y) - dot(v_dKstar_y.T, v) # (for y)
            dCov_s = dKstarstar_ds - dot(v.T, v_dKstar_s) - dot(v_dKstar_s.T, v)
            dCov_u = dKstarstar_du - dot(v.T, v_dKstar_u) - dot(v_dKstar_u.T, v)
            dCov_v = dKstarstar_dv - dot(v.T, v_dKstar_v) - dot(v_dKstar_v.T, v)
            dCov_w = dKstarstar_dw - dot(v.T, v_dKstar_w) - dot(v_dKstar_w.T, v)
            
            # (var)^-1 (-2 * Kstar^T (K + sigma^2 I)^-1 dKstar1)
            Dx = solve(Lvar.T, solve(Lvar, dCov_x)) 
            Dy = solve(Lvar.T, solve(Lvar, dCov_y))
            Ds = solve(Lvar.T, solve(Lvar, dCov_s))
            Du = solve(Lvar.T, solve(Lvar, dCov_u))
            Dv = solve(Lvar.T, solve(Lvar, dCov_v))
            Dw = solve(Lvar.T, solve(Lvar, dCov_w))
            
            # as per Seth's equations, we take the trace of these bad boys
            dx = trace(Dx) 
            dy = trace(Dy)
            ds = trace(Ds)
            du = trace(Du)
            dv = trace(Dv)
            dw = trace(Dw)
            
            ######################
            # now the second term  
            ######################
            
            dz_ds, dz_du, dz_dv, dz_dw = dRotate(q, Xnew[:,0], Xnew[:,1], Xnew[:,2])        
            # this extracts only the dz/dq components
            dz_ds, dz_du, dz_dv, dz_dw = dz_ds[:,2], dz_du[:,2], dz_dv[:,2], dz_dw[:,2]        
              
            # errors in z
            delt = z - mean
            
            # a lot packed in here...
            # -2 (z - mean)^T (var)^-1 dKstar^T (K + sigma^2 I)^-1 * y   
            dx -= -2*dot(delt.T,solve(Lvar.T, solve(Lvar, dot(dKstar_x.T, alpha))))
            dy -= -2*dot(delt.T,solve(Lvar.T, solve(Lvar, dot(dKstar_y.T, alpha))))
            ds -=  2*dot(delt.T,solve(Lvar.T, solve(Lvar, dz_ds - dot(dKstar_s.T, alpha))))
            du -=  2*dot(delt.T,solve(Lvar.T, solve(Lvar, dz_du - dot(dKstar_u.T, alpha))))
            dv -=  2*dot(delt.T,solve(Lvar.T, solve(Lvar, dz_dv - dot(dKstar_v.T, alpha))))
            dw -=  2*dot(delt.T,solve(Lvar.T, solve(Lvar, dz_dw - dot(dKstar_w.T, alpha))))
            
            # even more here...
            # - (z - mean)^T (var)^-1 (-2 * Kstar^T (K + sigma^2 I)^-1 dKstar1) (var)^-1 (z - mean)
            dx -= dot(dot(delt.T, Dx), solve(Lvar.T, solve(Lvar, delt)))
            dy -= dot(dot(delt.T, Dy), solve(Lvar.T, solve(Lvar, delt)))
            ds -= dot(dot(delt.T, Ds), solve(Lvar.T, solve(Lvar, delt)))
            du -= dot(dot(delt.T, Du), solve(Lvar.T, solve(Lvar, delt)))
            dv -= dot(dot(delt.T, Dv), solve(Lvar.T, solve(Lvar, delt)))
            dw -= dot(dot(delt.T, Dw), solve(Lvar.T, solve(Lvar, delt)))
            
            # z is super easy...
            Dz = ones((mean.shape[0],1))
            v1 = solve(Lvar, delt) # L^-1 (z - mean)
            v2 = solve(Lvar, Dz)   # L^-1 (1's)
            
            # (z - mean)^T (cov)^-1 1's 
            # (the one's vector just sums it up)
            dz = dot(v1.T, v2)[0]
            
        N = z.shape[0]

        def check_transform(stepX=0,stepY=0,stepZ=0,s=0,u=0,v=0,w=0):
            return transformPtsQ(Xnew, normalizeQ(q - array([s,u,v,w])), 
                                       array([tx-stepX,ty-stepY,tz-stepZ]))

        def check_likelihood(stepX=0,stepY=0,stepZ=0,s=0,u=0,v=0,w=0, cache=None):
            Xnew_transformed = transformPtsQ(Xnew, normalizeQ(q - array([s,u,v,w])), 
                                       array([tx-stepX,ty-stepY,tz-stepZ]))
            pts = Xnew_transformed[:,0:2]
            z = Xnew_transformed[:,2]        
            if cache:
                K, L, alpha = cache
                mean, var = gp(X, y, pts, K, L, alpha, sigma=sigma)
            else:
                mean, var = gp_chol(X, y, pts, sigma=sigma)
            return getLogL_chol(mean, var, z)

        delta = 1e-7
        cache = gp_bootstrap(X,y,sigma)
        l = check_likelihood(cache=cache)

        if not numerical:
            print "x %.05f vs %.05f"%(dx,(check_likelihood(stepX=-1 * delta, cache=cache) - l) / delta)
            print "y %.05f vs %.05f"%(dy,(check_likelihood(stepY=-1 * delta, cache=cache) - l) / delta)
            print "z %.05f vs %.05f"%(dz,(check_likelihood(stepZ=-1 * delta, cache=cache) - l) / delta)
            print "s %.05f vs %.05f"%(ds,(check_likelihood(s=-1 * delta, cache=cache) - l) / delta)
            print "u %.05f vs %.05f"%(du,(check_likelihood(u=-1 * delta, cache=cache) - l) / delta)
            print "v %.05f vs %.05f"%(dv,(check_likelihood(v=-1 * delta, cache=cache) - l) / delta)
            print "w %.05f vs %.05f"%(dw,(check_likelihood(w=-1 * delta, cache=cache) - l) / delta)
        
        if numerical:
                dx = (check_likelihood(stepX=-1 * delta) - l) / delta
                dy = (check_likelihood(stepY=-1 * delta) - l) / delta
                dz = (check_likelihood(stepZ=-1 * delta) - l) / delta
                ds = (check_likelihood(s=-1 * delta) - l) / delta
                du = (check_likelihood(u=-1 * delta) - l) / delta
                dv = (check_likelihood(v=-1 * delta) - l) / delta
                dw = (check_likelihood(w=-1 * delta) - l) / delta
                
                
#        import code; code.interact(local=locals())
        
        stepChange_t = beta/N#*sigma*sigma
        stepChange_q = beta * beta / N #stepChange_t ** 2 #beta/N#*sigma*sigma
        stepX, stepY, stepZ = stepChange_t*array([dx, dy, dz])
        stepQ = stepChange_q*array([ds, du, dv, dw]) 
        
        tx -= stepX
        ty -= stepY
        tz -= stepZ
        q = normalizeQ(q - stepQ) 
        
        Ts.append([tx, ty, tz, q])
        
        # now we transform the Xnew by q and tx,ty,tz for the next iteration
        Xnew_transformed = transformPtsQ(Xnew, q, array([tx,ty,tz]))
        pts = Xnew_transformed[:,0:2]
        z = Xnew_transformed[:,2]
        mean, var = gp_chol(X, y, pts, sigma=sigma)
        LL = getLogL_chol(mean, var, z)
        
        if verbose > 1:
            print '%5.4f\t'*8 % (tx, ty, tz, 
                                 q[0], q[1], q[2], q[3], 
                                 LL)
        if plotSteps:
            ax.clear()
            ax.plot(X[:,0],X[:,1],y,'g.')
            ax.plot(Xnew[:,0],Xnew[:,1],Xnew[:,2],'b.')
            ax.plot(Xnew_transformed[:,0],Xnew_transformed[:,1],Xnew_transformed[:,2],'r.')
            #pl.draw()

        if abs((LL-LL_last)/LL_last) < 1e-5: 
            break # or LL-LL_last > 0: break
            
        LL_last = LL
                

    likelihood = getLogL_chol(mean, var, z)
    if verbose > 0:
        print 'Stopped at iteration %d' % i
        print tx, ty, tz, likelihood

    if plotSteps:
        for i in range(len(Ts)):
            T = Ts[i]
            tx=T[0]
            ty=T[1]
            tz=T[2]
            q=T[3]

            Xnew_transformed = transformPtsQ(Xnew, q, array([tx,ty,tz]))
            ax.clear()
            ax.plot(X[:,0],X[:,1],y,'g.')
            ax.plot(Xnew[:,0],Xnew[:,1],Xnew[:,2],'b.')
            ax.plot(Xnew_transformed[:,0],Xnew_transformed[:,1],Xnew_transformed[:,2],'r.')
            pl.draw()
            import time
            time.sleep(0.01)
        time.sleep(10)
        
    if returnTraj:
        return q, [tx, ty, tz], likelihood, Ts
    else:    
        return q, [tx, ty, tz], likelihood 

        
    
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

def generate(X, Y, phi=0):
    import numpy as np
    R = 1 - np.sqrt(X**2 + Y**2)
    return np.cos(2 * np.pi * X + phi) * R + Y*Y 

def generate_sym(X, Y, phi=0):
    import numpy as np
    R = 1 - np.sqrt(X**2 + Y**2)
    return np.cos(2 * np.pi * X * Y + phi) * R 
    
def getSceneAndNew(randPoints=False, randPointsScene=False, T_vector = [0,1.5,-.5], generate=generate, n1 = 1000, n2=100,
                   qReal=array([1.0,0.0,0.0,0.0]), sigma=0.1):
    import numpy as np               
    # pick a whole bunch of random points (x,y) and then generate z + noise
    if randPointsScene:
        xs = 2*rand(n1) - 1
        ys = 2*rand(n1) - 1
        zs = generate(xs, ys, 0.0)
    else:
        xs, ys = np.meshgrid(np.linspace(-1, 1, sqrt(n1)), 
                             np.linspace(-1, 1, sqrt(n1)))
        zs = generate(xs, ys, 0.0)
   
    # now pick some new test points  
    if randPoints: 
        X = 2*rand(n2) - 1
        Y = 2*rand(n2) - 1
    else:
        X, Y = np.meshgrid(np.linspace(-1, 1, sqrt(n2)), 
                           np.linspace(-1, 1, sqrt(n2)))

    Z = generate(X, Y, 0.0)
    
    XYZ = np.concatenate([[X.flatten()],[Y.flatten()],[Z.flatten()]]).T
    XYZ_transformed = transformPtsQ(XYZ, qReal, T_vector)
    return (np.concatenate([[xs.flatten()],[ys.flatten()]]).T, zs.flatten(), 
            XYZ_transformed[:,:2],
            XYZ_transformed[:,2])
    
def case2D(plotIt=True, randPoints=False, T_vector = [0,1.5,-.5], generate=generate, n1 = 1000, n2=100,
           qReal=array([1.0,0.0,0.0,0.0]), sigma=0.1, randPointsScene=True):
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    from matplotlib import cm
        
    # pick a whole bunch of random points (x,y) and then generate z + noise
    if randPointsScene:
        xs = 2*rand(n1) - 1
        ys = 2*rand(n1) - 1
        zs = generate(xs, ys, 0.0)
    else:
        xs, ys = np.meshgrid(np.linspace(-1, 1, sqrt(n1)), 
                             np.linspace(-1, 1, sqrt(n1)))
        zs = generate(xs, ys, 0.0)
   
    # now pick some new test points  
    if randPoints == True: 
        X = 2*rand(n2) - 1
        Y = 2*rand(n2) - 1
    else:
        X, Y = np.meshgrid(np.linspace(-1, 1, sqrt(n2)), 
                           np.linspace(-1, 1, sqrt(n2)))

    Z = generate(X, Y, 0.0)

    # now predict the z component from the xs,ys,zs
    st = time.time()
    
    mean, var = gp_chol(np.concatenate([[xs.flatten()],[ys.flatten()]]).T, zs.flatten(), 
                        np.concatenate([[X.flatten() ],[Y.flatten() ]]).T, sigma=.001)
    #print 'time chol:', time.time() - st; st = time.time()
    #mean2, var2 = gp_simple(np.concatenate([[xs],[ys]]).T, zs, np.concatenate([[X.flatten()],[Y.flatten()]]).T, sigma=.001)
    #print 'time simple:', time.time() - st
        
    
    if plotIt == True:    
        '''fig = plt.figure()
        plt.plot(linspace(-1,1,100), [getLogL_chol(mean, var, Z.flatten()+i) for i in linspace(-1,1,100)]) 
        plt.title('Varying $t_z$ from -1 to 1')
        plt.ylabel('Negative Log-Likelihood')
        plt.xlabel('$t_z$')'''
    
        '''fig = plt.figure()

        var_masked = var.copy()
        var_masked[var <= 0] = 1e-10
        plt.imshow(shuffleIt(log(var_masked)), interpolation='nearest')''' 
    
        fig = plt.figure()

        #ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        
        
        
        xs2 = np.linspace(-1, 1, 50)
        ys2 = np.linspace(-1, 1, 50)
        Xs, Ys = np.meshgrid(xs2, ys2)
        Zs = generate(Xs, Ys, 0.0)
        
        print dir(cm)

        ax.plot_surface(Xs, Ys, Zs, linewidth=0, cmap=cm.Blues, alpha=0.5, rstride=1, cstride=1, label='Actual Surface')
        #wframe = ax.plot_wireframe(Xs, Ys, Zs, rstride=1, cstride=1)
        
        scene = ax.scatter(xs, ys, zs, c='r', s=50, label='Point Cloud')
        
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-0, 1)
        
        #ax.title('Actual Surface and Point Cloud')
        
        
        fig = plt.figure()

        #ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
   
        predicted = ax.plot_surface(X, Y, mean.reshape(X.shape), linewidth=0, cmap=cm.hot, alpha=0.8, rstride=1, cstride=1, label='Reconstruction')
        #predicted = ax.plot_wireframe(X, Y, mean.reshape(X.shape), color='g')

        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-0, 1)

        #ax.legend(['Reconstruction'])
        #ax.title('Reconstruction')
        
        plt.show()
    
    XYZ = np.concatenate([[X.flatten()],[Y.flatten()],[Z.flatten()]]).T
    XYZ_transformed = transformPtsQ(XYZ, qReal, T_vector)
    return (np.concatenate([[xs.flatten()],[ys.flatten()]]).T, zs.flatten(), 
            XYZ_transformed[:,:2],
            XYZ_transformed[:,2])

def test():
    sigma = 0.1

    T_vector = [0.2,0.2,-.2] #.075,-.02,.03]
    qReal = Q.rotate('Z', vectors.radians(-30)) * Q.rotate('X', vectors.radians(15)) #-5))
    u, v, w, s = qReal
    qReal = array([s, u, v, w])
    print qReal
    print 'Translation', T_vector
    X,y,pts,z = case2D(randPoints=False, randPointsScene=True, plotIt=True, 
                       T_vector=T_vector, qReal=qReal, n1=100, n2=50*50, 
                       sigma=sigma)
#    X2,y2,pts2,z2 = case2D(randPoints=True, plotIt=False, 
#                       T_vector=T_vector, qReal=qReal, n1=200, n2=100, 
#                       sigma=sigma)
#    import numpy as np
#    X=np.concatenate([X,X2],0)
#    y=np.concatenate([y,y2],0)
#    pts=np.concatenate([pts,pts2],0)
#    z=np.concatenate([z,z2],0)

    #print X,y,pts,z
    #exit()
    
    raw_input()
    
    q, t, LL = gradientDescent(X,y,pts,z,sigma=sigma, iterations=200, beta=0.01)
    print 'Real Translation', around(T_vector, decimals=2)
    print 'Real Rotation'
    print around(rotQ(qReal), decimals=2)
    print 'Calculated t', around(-dot(inv(rotQ(q)),t), decimals=2)
    print 'Calculated R'
    print around(linalg.inv(rotQ(q)), decimals=2)
    
if __name__ == '__main__':
    #case1D()

    test()
