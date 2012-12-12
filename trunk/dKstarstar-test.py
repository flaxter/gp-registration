from __future__ import division
from numpy import *
from numpy.linalg import norm, inv, cholesky, solve, det
from numpy.random import randn, rand
import time
from scipy.spatial.distance import cdist, pdist
from quaternions import dRotate, normalizeQ, transformPts, rotQ
import vectors
from vectors import M, Q
import numpy as np

from gp import generate

def getDKstarstar(Xp, Xnew, q, k1=1, k2=1):
    # reeeallly subtle broadcasting here
    # in matlab I would do a diag(X)*K*diag(Y) sort of thing
    print Xp.shape
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
    
def getDKstarstar2(Xp, Xnew, q, k1=1, k2=1):
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
    
if __name__ == '__main__':
    n2 = 100
 
    X = 2*rand(n2) - 1
    Y = 2*rand(n2) - 1
    Z = generate(X, Y)    

    XYZ = np.concatenate([[X.flatten()],[Y.flatten()],[Z.flatten()]]).T
    XYZ_transformed, t, q = transformPts(XYZ, [10,10,10], [.05,.5,.2])
    
    st = time.time()
    dk_s, dk_u, dk_v, dk_w = getDKstarstar(XYZ_transformed[:,:2], XYZ, [0.0,-1.0,0.0,0.0])
    print time.time() - st; st = time.time()
    
    dk_s2, dk_u2, dk_v2, dk_w2 = getDKstarstar2(XYZ_transformed[:,:2], XYZ, [0.0,-1.0,0.0,0.0])
    print time.time() - st
    
    print abs(dk_s2 - dk_s) 
    
