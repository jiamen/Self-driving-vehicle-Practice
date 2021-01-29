
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, Matrix
from sympy.interactive import printing
# sympy是一个Python的科学计算库，用一套强大的符号计算体系完成诸如多项式求值、求极限、解方程、求积分、微分方程、级数展开、矩阵运算等等计算问题。虽然Matlab的类似科学计算能力也很强大。



## 初始化行人状态 x = (p_x, p_y, v_x, v_y)^T
x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T     # slamP240页 程序运行期间需要维护这个状态量，对它不断进行更新迭代
print('x:\n', x, x.shape)
## 预测误差，先验估计协方差矩阵
P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])
print('P:\n', P, P.shape)

dt = 0.1    # Time Step between Filter Steps    测量的时间间隔

## 状态转移矩阵
F = np.matrix([[1.0, 0.0, dt, 0.0],
               [0.0, 1.0, 0.0, dt],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])
print('F:\n', F, F.shape)
## 观测矩阵，为了维持维度统一
H = np.matrix([[0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])
print('H:\n', H, H.shape)

## 测量噪声的协方差矩阵R
ra = 10.0 ** 2
R = np.matrix([[ra, 0.0],
               [0.0, ra]])
print('R:\n', R, R.shape)
ra = 0.09
R = np.matrix([[ra, 0.0],
               [0.0, ra]])
print('R:\n', R, R.shape)


## 过程噪声的协方差矩阵Q
sv = 0.5    # 行人速度
G = np.matrix([[0.5*dt**2],
               [0.5*dt**2],
               [dt],
               [dt]])
Q = G*G.T*sv**2
print('Q1:\n', Q, Q.shape)
# 过程噪声Q的另一种表示方法
printing.init_printing()
dts = Symbol('dt')
Qs = Matrix([[0.5*dts**2], [0.5*dts**2], [dts], [dts]])
Qs = Qs*Qs.T
print('Q2:\n', Qs, Qs.shape)


## 定义1个单位矩阵
I = np.eye(4)
print('I:\n', I, I.shape)


## 随机产生一些测量数据， 产生一些随机测量数据阵
m = 200     # Measurements, 200个测量数据
vx = 20     # in X
vy = 10     # in Y

## 得到mx和my两个实际速度观测量，也就是最初定义的x=(p_x, p_y, v_x, v_y)^T中的后两维v_x,v_y的实际观测值，前两维位置p_x, p_y无法通过观测得到
mx = np.array(vx+np.random.randn(m))
my = np.array(vy+np.random.randn(m))
# print('mx.shape: ', mx.shape)             # (200,)
measurements = np.vstack( (mx, my) )        # (2, 200)   按垂直方向（行顺序）堆叠数组构成一个新的数组

print('measurements.shape: ', measurements.shape)
print('Standard Deviation of Acceleration Measurement = %.2f ' % np.std(mx))
print('You assumed %.2f in R.' % R[0, 0])


fig = plt.figure( figsize=(12, 4) )         # 图片大小,这里准备显示的是状态量x中的后两维v_x,v_y的实际测量值m_x,m_y
plt.step(range(m), mx, label='$\dot m\dot x$')    # 横坐标，纵坐标，图标
plt.step(range(m), my, label='$\dot m\dot y$')
plt.ylabel(r'Veclocity $m/s$')              # 纵坐标 标识
plt.title('Measurements')                   # 图片标题
plt.legend(loc='best', prop={'size': 16})   # 设置图例边框，就是上面的 x点 和 y点
plt.show()


## 一些过程值，用于结果的显示：
xt  = []    # 状态量x的第1维 位置p_x
yt  = []    # 位置p_y
dxt = []    # 状态量x的第3维 速度v_x
dyt = []    # 速度v_y

Zx  = []    # 实际速度观测值m_x
Zy  = []    # m_y
Px  = []    # 预测误差
Py  = []
Pdx = []
Pdy = []
Rdx = []    # 测量噪声的协方差矩阵R
Rdy = []
Kx  = []    # 卡尔曼增益 K 第一维
Ky  = []
Kdx = []
Kdy = []

def savestates(x, Z, P, R, K):
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    dxt.append(float(x[2]))
    dyt.append(float(x[3]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Px.append(float(P[0, 0]))
    Py.append(float(P[1, 1]))
    Pdx.append(float(P[2, 2]))
    Pdy.append(float(P[3, 3]))
    Rdx.append(float(R[0, 0]))
    Rdy.append(float(R[1, 1]))
    Kx.append(float(K[0, 0]))
    Ky.append(float(K[1, 0]))
    Kdx.append(float(K[2, 0]))
    Kdy.append(float(K[3, 0]))


# print('len(measurements[0]): ', len(measurements[0]), '\n')         # 第1个 (1, 100)
# print('len(measurements[1]): ', len(measurements[1]), '\n')         # 第2个 (1, 100)
for n in range(len(measurements[0])):
    # Time Update (Prediction)
    # ========================
    # Project the state ahead               （1）状态预测，F是状态转移矩阵
    x = F*x

    # Project the error covariance ahead    （2）计算预测误差
    P = F*P*F.T + Q     # print(P.shape) 4×4


    # Measurement Update (Correction)
    # ================================
    # Compute the Kalman Gain               （3）计算卡尔曼增益K
    S = H*P*H.T + R     # 无人驾驶课本P78页，分母   H是观测矩阵，为了维持维度一致，维持运算，通过H将4×4矩阵P变为2×2，能够与测量噪声的2×2协方差矩阵R 相加
    K = (P*H.T)*np.linalg.pinv(S)       # np.linalg.pinv(S)是S^(-1)


    # Update the estimate via z             （4）更新状态量
    Z = measurements[:, n].reshape(2, 1)    # 得到观测量
    y = Z - (H*x)
    x = x + (K*y)       # 通过观测量Z 更新 始终维护的状态量x

    # Update the error covariance       更新预测误差协方差
    P = (I - (K*H)) * P

    # Save states (for Plotting)
    savestates(x, Z, P, R, K)


def plot_x():
    figure = plt.figure(figsize=(16, 9))
    plt.step(range(len(measurements[0])), dxt, label='$estimateVx$')        # 显示状态量估计值v_x, v_y
    plt.step(range(len(measurements[1])), dyt, label='$estimateVy$')

    plt.step(range(len(measurements[0])), measurements[0], label='$measurementVx$')     # 显示上文已经显示过的实际测量值m_x, m_y做对比
    plt.step(range(len(measurements[1])), measurements[1], label='$measurementVy$')

    plt.axhline(vx, color='#999999', label='$trueVx$')          # 灰色显示实际值vx, vy
    plt.axhline(vy, color='#999999', label='$trueVy$')

    plt.xlabel('Filter Step')   # 指定横纵坐标名称
    plt.ylabel('Velocity')
    plt.ylim([0, 30])           # 对y坐标值进行范围限定
    plt.title('Estimate (Estimate from State Vector $x$)')
    plt.legend(loc='best', prop={'size': 11})       # 设置图例边框属性
    plt.show()

plot_x()


def plot_xy():
    figure = plt.figure(figsize=(12, 12))
    plt.scatter(xt, yt, s=20, label='State', c='k')             # s=20, 大小；c='k' 黑色     这是数组内所有点
    plt.scatter(xt[0], yt[0], s=100, label='Start', c='g')      # c='g' 绿色      这只是一个点
    plt.scatter(xt[-1], yt[-1], s=100, label='Goal', c='r')     # c='r' 红色      这只是一个点

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position')
    plt.legend(loc='best')
    plt.axis('equal')
    plt.show()

plot_xy()
