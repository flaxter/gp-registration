from numpy import *
from numpy.linalg import cholesky, inv, solve
from numpy.random import randn
import time

A = randn(5000*5000)
A = A.reshape((5000,5000))
A = dot(A,A.T)
'''st = time.time()
inv1 = inv(A)
print time.time() - st'''

L = cholesky(A)

y = randn(5000)

st = time.time()
#inv2 = dot(inv(L).T,inv(L))
inv2 = solve(L.T,solve(L,y))
print time.time() - st

'''
diff = abs(inv1 - inv2)

print diff[diff > 1e-8]
'''
