
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
import numdifftools as nd           # numdifftools 库是一套用python编写的工具 求解一个或多个变量的自动数值微分问题。
import math


dataset = []

# read the measurement data, use 0.0 to stand LIDAR data and 1.0 stand RADAR data



