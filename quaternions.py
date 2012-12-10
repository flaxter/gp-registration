from numpy import *
import vectors
from vectors import M, Q
                  
def rotQ(q):
    u, v, w, s = q

    return array([[s*s+u*u-v*v-w*w, 2*(u*v-s*w), 2*(u*w+s*v)],
                  [2*(u*v+s*w), s*s-u*u+v*v-w*w, 2*(v*w-s*u)],
                  [2*(u*w-s*v), 2*(v*w+s*u), s*s-u*u-v*v+w*w]])
    
def skew(x):    
    return array([[      0,  x[2], -x[1]],
                  [-x[2],       0,  x[0]],
                  [ x[1],   -x[0],     0]])

def dRotate(q, x, y, z):
    u, v, w, s = q
    ds = [  u*x + v*y + w*z , 
            v*x - u*y - s*z , 
            w*x + s*y - u*z ]
                           
    du = [ -v*x + u*y + s*z , 
            u*x + v*y + w*z , 
           -s*x + w*y - v*z ] 
                           
    dv = [ -w*x - s*y + u*z ,  
            s*x - w*y + v*z ,  
            u*x + v*y + w*z ]
                           
    dw = [  s*x - w*y + v*z ,
            w*x + s*y - u*z ,
           -v*x + u*y + s*z ]
    return ds, du, dv, dw     
    
def multQ(q,p):
    a1, b1, c1, d1 = q
    a2, b2, c2, d2 = p
    return array([-a1*a2+b1*b2+c1*c2+d1*d2,
                   a1*d2+b1*c2-c1*b2+d1*a2,
                   a1*c2-b1*d2+c1*a2+d1*b2,
                   a1*b2+b1*a2+c1*d2-d1*c2])
                  
def normalize(q):
    return q/sqrt(dot(q,q))             
                  
def transformPts(pts, rot=[5,-5,5], trans=[0.1,-.1,.1]):  
    qReal = Q.rotate('X', vectors.radians(rot[0]))* \
            Q.rotate('Y', vectors.radians(rot[1]))* \
            Q.rotate('Z', vectors.radians(rot[2]))
    tReal = array(trans)
    
    # transform sample by the unknown R, t
    pts = dot(rotQ(qReal), pts.T).T + tReal

    return pts, tReal, qReal

