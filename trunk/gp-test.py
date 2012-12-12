from __future__ import division
from numpy import *
from numpy.linalg import norm, inv, cholesky, solve, det
from numpy.random import randn, rand
import time
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from quaternions import dRotate
from gp import gradientDescent
 
def generate(X,Y,phi):
    return X

def applyTransform(D,T):
    import numpy as np
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
	Tx.append((rand()*0.3,0,0))

    Ty=[]
    for i in range(5):
	Ty.append((0,rand()*0.3,0))

    Txy=[]
    for i in range(5):
	Txy.append((rand()*0.3,rand()*0.3,0))

    Tz=[]
    for i in range(5):
	Tz.append((0,0,rand()*0.3))

    Txyz=[]
    for i in range(5):
	Txyz.append((rand()*0.3,rand()*0.3,rand()*0.3))

    T=[]
    T.extend(Ty)
    T.extend(Tx)
    T.extend(Txy)
    T.extend(Tz)
    T.extend(Txyz)
    return T

def generateSmooth(X,Y,phi):
    import numpy as np
    R = 1 - np.sqrt(X**2 + Y**2)
    Z = np.cos(2 * np.pi * X/3 * Y/3 + phi) * R
    return Z

def generateAbs(X,Y,phi):
    import numpy as np
    Z=abs(X-mean(X))+abs(Y-mean(Y))+phi
    return Z

def generatePlanes(numSurfs,n,p):
    import numpy as np
    X=np.zeros(0)
    Y=np.zeros(0)
    Z=np.zeros(0)
    for i in range(numSurfs):
	x,y=(rand(2)-0.5)*0.5
	w,h=(rand(2)-0.5)*0.3
	phi=rand(1)*0.1
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

def gd(X,y,pts,z,sigma=0.1,verbose=2,iterations=35,beta=0.0005):
	print X.shape
	print y.shape
	print pts.shape
	print z.shape
	print sigma,verbose,iterations
	return gradientDescent(X,y,pts,z,sigma=sigma,verbose=verbose,iterations=iterations,beta=beta)

if True:
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
	    ax.clear()
            ax.plot(D[0],D[1],D[2],'b.')
            ax.plot(TD[0],TD[1],TD[2],'r.')
#	    pl.savefig('gp-dataset.png')
#            pl.show()
#            break
##           Pass in D,TD to the GP function, return THAT
	    print delta
            sigma=delta*10
#            import code; code.interact(local=locals())
	    
            for beta in [0.001,0.01,0.1,1,10]:
            	q,t,l = gd(array(D[0:2]).T,D[2],array(TD[0:2]).T,TD[2],sigma=sigma,verbose=2,iterations=500,beta=beta)
##           Compare to T, evaluate accuracy
            print T
            print t
#            print t-T,sqrt((t[0]-T[0])**2+(t[1]-T[1])**2+(t[2]-T[2])**2)
            print
#

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

