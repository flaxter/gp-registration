from numpy import *
import vectors
from vectors import M, Q
                  
def rotQ(q):
    s, u, v, w = q

    return array([[s*s+u*u-v*v-w*w, 2*(u*v-s*w), 2*(u*w+s*v)],
                  [2*(u*v+s*w), s*s-u*u+v*v-w*w, 2*(v*w-s*u)],
                  [2*(u*w-s*v), 2*(v*w+s*u), s*s-u*u-v*v+w*w]])
    
def skew(x):    
    return array([[      0,  x[2], -x[1]],
                  [-x[2],       0,  x[0]],
                  [ x[1],   -x[0],     0]])

def dRotate(q, x, y, z):
    s, u, v, w = q
    du = 2*array([  u*x + v*y + w*z , 
                    v*x - u*y - s*z , 
                    w*x + s*y - u*z ])
                           
    dv = 2*array([ -v*x + u*y + s*z , 
                    u*x + v*y + w*z , 
                   -s*x + w*y - v*z ])
                           
    dw = 2*array([ -w*x - s*y + u*z ,  
                    s*x - w*y + v*z ,  
                    u*x + v*y + w*z ])
                           
    ds = 2*array([  s*x - w*y + v*z ,
                    w*x + s*y - u*z ,
                   -v*x + u*y + s*z ])
           
    # so ds is a three vector that represents dx/ds, dy/ds, dz/ds
    return array(ds).T, array(du).T, array(dv).T, array(dw).T     
    
def multQ(q,p):
    a1, b1, c1, d1 = q
    a2, b2, c2, d2 = p
    return array([-a1*a2+b1*b2+c1*c2+d1*d2,
                   a1*d2+b1*c2-c1*b2+d1*a2,
                   a1*c2-b1*d2+c1*a2+d1*b2,
                   a1*b2+b1*a2+c1*d2-d1*c2])
                  
def normalizeQ(q):
    return q/sqrt(dot(q,q))             
                  
def transformPts(pts, rot=[5,-5,5], trans=[0.1,-.1,.1]):  
    qReal = Q.rotate('Z', vectors.radians(rot[2]))* \
            Q.rotate('Y', vectors.radians(rot[1]))* \
            Q.rotate('X', vectors.radians(rot[0]))
    u, v, w, s = qReal
    qReal = array([s, u, v, w])
    tReal = array(trans)
    
    # transform sample by the unknown R, t
    pts2 = dot(rotQ(qReal), pts.T).T + tReal
    return pts2, tReal, qReal
    
def getQ(rot=[5,-5,5]):
    qReal = Q.rotate('Z', vectors.radians(rot[2]))* \
            Q.rotate('Y', vectors.radians(rot[1]))* \
            Q.rotate('X', vectors.radians(rot[0]))
    u, v, w, s = qReal
    return array([s, u, v, w])

def transformPtsQ(pts, q, t):  
    # transform sample by the unknown R, t
    t = array(t)
    pts2 = dot(rotQ(q), pts.T).T + t
    return pts2
    
if __name__ == '__main__':
    from numpy.random import randn
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    import matplotlib.pyplot as plt
    
    pts = randn(3*200).reshape((200,3))
    
    fig = plt.figure()

    #ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)

    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='g')
       
    pts2 = transformPtsQ(pts, array([1,0,0,0]), [2.5,2.5,2.5])
    
    ax.scatter(pts2[:,0], pts2[:,1], pts2[:,2], c='b')

    #ax.set_xlim3d(-1, 1)
    #ax.set_ylim3d(-1, 1)
    #ax.set_zlim3d(-1, 1)

    plt.show()
    
    
