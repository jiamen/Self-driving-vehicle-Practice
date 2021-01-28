
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T
print('x:\n', x, x.shape)


P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])
print('P:\n', P, P.shape)


dt = 0.1    # Time Step between Filter Steps
F = np.matrix([[1.0, 0.0, dt, 0.0],
               [0.0, 1.0, 0.0, dt],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])
print('F:\n', F, F.shape)


H = np.matrix([[0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])
print('H:\n', H, H.shape)

ra = 10.0 ** 2
R = np.matrix([[ra, 0.0],
               [0.0, ra]])
print('R:\n', R, R.shape)


ra = 0.09
R = np.matrix([[ra, 0.0],
               [0.0, ra]])
print('R:\n', R, R.shape)

sv = 0.5
G = np.matrix([[0.5*dt**2],
               [0.5*dt**2],
               [dt],
               [dt]])
Q = G*G.T*sv**2
print('Q1:\n', Q, Q.shape)

from sympy import Symbol, Matrix
from sympy.interactive import printing
# sympy是一个Python的科学计算库，用一套强大的符号计算体系完成诸如多项式求值、求极限、解方程、求积分、微分方程、级数展开、矩阵运算等等计算问题。虽然Matlab的类似科学计算能力也很强大。

printing.init_printing()
dts = Symbol('dt')
Qs = Matrix([[0.5*dts**2], [0.5*dts**2], [dts], [dts]])
Qs = Qs*Qs.T
print('Q2:\n', Qs, Qs.shape)


I = np.eye(4)
print(I, I.shape)



