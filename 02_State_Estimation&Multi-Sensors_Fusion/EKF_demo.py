
from __future__ import print_function
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.stats import norm
# Scipy是一个用于数学、科学、工程领域的常用软件包，可以处理插值、积分、优化、图像处理、常微分方程数值解的求解、信号处理等问题。
# 它用于有效计算Numpy矩阵，使Numpy和Scipy协同工作，高效解决问题。
from sympy import Symbol, symbols, Matrix, sin, cos, sqrt, atan2
from sympy import init_printing
init_printing(use_latex=True)       # 对sympy库的输出结果进行LaTeX渲染
import numdifftools as nd           # numdifftools 库是一套用python编写的工具 求解一个或多个变量的自动数值微分问题。本例中使用该库直接计算雅克比矩阵。
import math


## 读取数据集
dataset = []
# read the measurement data, use 0.0 to stand LIDAR data and 1.0 stand RADAR data.
with open('data_synthetic.txt', 'rb') as f:
    lines = f.readlines()       # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，该列表可以由 Python 的 for... in ... 结构进行处理。
                                # 如果碰到结束符 EOF 则返回空字符串。
    for line in lines:
        line = line.decode().strip('\n')
        line = line.strip()     # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        numbers = line.split()
        result = []
        for i, item in enumerate(numbers):
            item.strip()        # i是字符序号，item是具体的字符
            if i == 0:          # 开头用1.0和0.0 区分是 Lidar还是Radar
                if item == 'L':
                    result.append(0.0)
                else:
                    result.append(1.0)
            else:
                result.append(float(item))
        dataset.append(result)
    f.close()       # 一定记得关闭文件


## 初始化数据 并 做处理， 初始化P、激光雷达的测量矩阵(线性)H_L, 测量噪声R,
## 以及过程噪声中的直线加速度项的标准差σ_a, 转角加速度项的标准差σ_w
P = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])
print('P: \n', P, P.shape)
H_lidar = np.array([[1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.]])
print('H_lidar: \n', H_lidar, H_lidar.shape)

R_lidar = np.array([[0.0225, 0.], [0., 0.0225]])
R_radar = np.array([[0.09, 0., 0.], [0., 0.0009, 0.], [0., 0., 0.09]])
print('R_lidar: \n', R_lidar, R_lidar.shape)
print('R_radar: \n', R_radar, R_radar.shape)


# process noise standard deviation for a
std_noise_a = 2.0
# process noise standard deviation for yaw acceleration.
std_noise_yaw_dd = 0.3

## 在整个预测和预测更新过程中，所有角度的测量数值都应该控制在[-π, π]，由于角度加减2π保持不变，用如下函数调整角度
def control_psi(psi):
    while psi > np.pi or psi < -np.pi:
        if psi > np.pi:
            psi = psi - 2 * np.pi
        if psi < -np.pi:
            psi = psi + 2 * np.pi
    return psi


## 数据初始化代码, 这里的state就是x， x(t)=(x, y, v, θ, w), 只初始化前2位
state = np.zeros(5)
init_measurement = dataset[0]
current_time = 0.0
if init_measurement[0] == 0.0:  # 若使用第一个激光雷达的测量数据初始化对象状态
    print('Initialize with LIDAR measurement!')
    current_time = init_measurement[3]
    state[0] = init_measurement[1]              # 直接将测量到的目标的(x, y)坐标作为初始坐标, 其余状态初始化为0
    state[1] = init_measurement[2]
else:
    print('Initialize with RADAR measurement!')
    current_time = init_measurement[4]
    init_rho = init_measurement[1]              # 目标车辆在车辆坐标系下与本车的极坐标距离ρ
    init_psi = init_measurement[2]              # 目标车辆与x轴的夹角Ψ
    init_psi = control_psi(init_psi)            # 将测量值控制在[-π, π]
    state[0] = init_rho * np.cos(init_psi)
    state[1] = init_rho * np.sin(init_psi)

print('state： ', state, state.shape)


## 写一个辅助函数用于保存数值 Preallocation for Saving
px = []
py = []
vx = []
vy = []

gpx = []
gpy = []
gvx = []
gvy = []

mx = []
my = []

def savestates(ss, gx, gy, gv1, gv2, m1, m2):
    px.append(ss[0])
    py.append(ss[1])
    vx.append(np.cos(ss[3]) * ss[2])
    vy.append(np.sin(ss[3]) * ss[2])

    gpx.append(gx)
    gpy.append(gy)
    gvx.append(gv1)
    gvy.append(gv2)
    mx.append(m1)
    my.append(m2)


## 定义状态转移函数和测量函数，使用numdifftools库来计算其对应的雅克比矩阵
measurement_step = len(dataset)
state = state.reshape([5, 1])
dt = 0.05                       # △t=0.5, 这里是定值，实际运行EKF时需要计算前后两次测量的时间差来替换这里的△t

I = np.eye(5)

## 根据w是否为0，区分运动模型，进一步区分状态转移函数
# when omega is not 0       根据《无人驾驶》课本P86页公式编写
transition_function = lambda y: np.vstack( (
    y[0] + (y[2] / y[4]) * (np.sin(y[3] + y[4] * dt) - np.sin(y[3])),       # f1
    y[1] + (y[2] / y[4]) * (-np.cos(y[3] + y[4] * dt) + np.cos(y[3])),      # f2
    y[2],                                                                   # f3
    y[3] + y[4] * dt,                                                       # f4
    y[4]                                                                    # f5
) )

# when omega is 0
transition_function_1 = lambda m: np.vstack( (
    m[0] + m[2] * np.cos(m[3]) * dt,
    m[1] + m[2] * np.sin(m[3]) * dt,
    m[2],
    m[3] + m[4] * dt,
    m[4]
) )

J_A   = nd.Jacobian(transition_function)        # J = (∂f/∂x_1, ..., ∂f/∂x_n)
J_A_1 = nd.Jacobian(transition_function_1)


# 将状态x = (x, y, v, θ, w)映射到 (ρ, Ψ, ρ`)
measurement_function = lambda k: np.vstack( (
    np.sqrt(k[0] * k[0] + k[1] * k[1]),
    math.atan2(k[1], k[0]),
    ( k[0] * k[2] * np.cos(k[3]) + k[1] * k[2] * np.sin(k[3]) ) / np.sqrt(k[0] * k[0] + k[1] * k[1])
))

J_H = nd.Jacobian(measurement_function)



## EKF 的过程代码   x = (x, y, v, θ, w)
for step in range(1, measurement_step):
    # Prediction
    # =====================
    t_measurement = dataset[step]
    if t_measurement[0] == 0.0:     # 是Lidar数据
        m_x = t_measurement[1]      # 测量目标得到的x位置
        m_y = t_measurement[2]      # 测量目标得到的y位置
        z = np.array( [[m_x], [m_y]] )      # 观测数组

        dt = (t_measurement[3] - current_time) / 1000000.0
        current_time = t_measurement[3]

        # true position 实际的(x, y, v_x, v_y)
        g_x = t_measurement[4]
        g_y = t_measurement[5]
        g_v_x = t_measurement[6]
        g_v_y = t_measurement[7]

    else:                           # 是Radar数据
        m_rho = t_measurement[1]
        m_psi = t_measurement[2]
        m_dot_rho = t_measurement[3]
        z = np.array([[m_rho], [m_psi], [m_dot_rho]])

        dt = (t_measurement[4] - current_time) / 1000000.0
        current_time = t_measurement[4]

        # true position
        g_x = t_measurement[5]
        g_y = t_measurement[6]
        g_v_x = t_measurement[7]
        g_v_y = t_measurement[8]

    if np.abs(state[4, 0]) < 0.0001:     # omega is 0, Driving straight
        state = transition_function_1(state.ravel().tolist())
        state[3, 0] = control_psi(state[3, 0])
        JA = J_A_1(state.ravel().tolist())
    else:                               # otherwise
        state = transition_function(state.ravel().tolist())
        state[3, 0] = control_psi(state[3, 0])
        JA = J_A(state.ravel().tolist())

    G = np.zeros([5, 2])                # 这是课本第88页 noise_term 过程噪声
    G[0, 0] = 0.5 * dt * dt * np.cos(state[3, 0])
    G[1, 0] = 0.5 * dt * dt * np.sin(state[3, 0])
    G[2, 0] = dt
    G[3, 1] = 0.5 * dt * dt
    G[4, 1] = dt

    Q_v = np.diag([std_noise_a*std_noise_a, std_noise_yaw_dd*std_noise_yaw_dd])     # E[uu^T]
    Q = np.dot(np.dot(G, Q_v), G.T)             # 得到过程噪声

    # Project the error covariance ahead
    P = np.dot(np.dot(JA, P), JA.T) + Q         # 计算预先规划误差协方差

    # Measurement Update (Correction)
    # ===============================
    if t_measurement[0] == 0.0:
        # Lidar
        S = np.dot(np.dot(H_lidar, P), H_lidar.T) + R_lidar         # 这里的H_lidar是观测方程
        K = np.dot(np.dot(P, H_lidar.T), np.linalg.inv(S))          # 计算卡尔曼增益

        # 得到卡尔曼增益后准备更新状态估计和预测误差协方差
        y = z - np.dot(H_lidar, state)              # state = x = (x, y, v, θ, w)
        # y[1, 0] = control_psi(y[1, 0])
        state = state + np.dot(K, y)                # 通过测量z更新状态估计state
        state[3, 0] = control_psi(state[3, 0])
        # Update the error covariance
        P = np.dot((I - np.dot(K, H_lidar)), P)     # 更新误差协方差

        # Save states for Plotting
        savestates(state.ravel().tolist(), g_x, g_y, g_v_x, g_v_y, m_x, m_y)

    else:
        # Radar
        JH = J_H(state.ravel().tolist())

        S = np.dot(np.dot(JH, P), JH.T) + R_radar       # 这是P90页的公式的分母，针对radar来说
        K = np.dot(np.dot(P, JH.T), np.linalg.inv(S))   # 这是卡尔曼增益K
        map_pred = measurement_function(state.ravel().tolist())     # 将状态x = (x, y, v, θ, w)映射到 (ρ, Ψ, ρ`)
        if np.abs(map_pred[0, 0]) < 0.0001:
            # if rho is 0
            map_pred[2, 0] = 0

        y = z - map_pred
        y[1, 0] = control_psi(y[1, 0])      # 这个是第二个元素Ψ，需要控制在[-π, π]

        state = state + np.dot(K, y)
        state[3, 0] = control_psi(state[3, 0])
        # Update the error covariance
        P = np.dot((I - np.dot(K, JH)), P)

        savestates(state.ravel().tolist(), g_x, g_y, g_v_x, g_v_y, m_rho*np.cos(m_psi), m_rho*np.sin(m_psi))    # 最后两个是极坐标转换


def rmse(estimates, actual):
    result = np.sqrt( np.mean((estimates-actual)**2) )
    return result

print(rmse(np.array(px), np.array(gpx)),
      rmse(np.array(py), np.array(gpy)),
      rmse(np.array(vx), np.array(gvx)),
      rmse(np.array(vy), np.array(gvy)) )


# write to the output file
stack = [px, py, vx, vy, mx, my, gpx, gpy, gvx, gvy]
stack = np.array(stack)
stack = stack.T
np.savetxt('output.csv', stack, '%.6f')


import plotly.offline as py
from plotly.graph_objs import *
import pandas as pd
import math

# py.init_notebook_mode()

my_cols = ['px_est', 'py_est', 'vx_est', 'vy_est', 'px_meas', 'py_meas', 'px_gt', 'py_gt', 'vx_gt', 'vy_gt']
with open('output.csv') as f:
    table_ekf_output = pd.read_table(f, sep=' ', header=None, names=my_cols, lineterminator='\n')

    # table_ekf_output

# Measurements
trace2 = Scatter(
    x=table_ekf_output['px_meas'],
    y=table_ekf_output['py_meas'],
    xaxis='x2',
    yaxis='y2',
    name='Measurements',
    mode='markers'
)

# Estimations
trace1 = Scatter(
    x=table_ekf_output['px_est'],
    y=table_ekf_output['py_est'],
    xaxis='x2',
    yaxis='y2',
    name='KF - Estimate',
    mode='markers'
)


# Ground Truth
trace3 = Scatter(
    x=table_ekf_output['px_gt'],
    y=table_ekf_output['py_gt'],
    xaxis='x2',
    yaxis='y2',
    name='Ground Truth',
    mode='markers'
)

data = [trace1, trace2, trace3]

layout = Layout(
    xaxis2=dict(anchor='x2', title='px'),
    yaxis2=dict(anchor='y2', title='py')
)

fig = Figure(data=data, layout=layout)
py.plot(fig, filename='EKF.html')

