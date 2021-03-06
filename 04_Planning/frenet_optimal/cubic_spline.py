
import math
import numpy as np
import bisect
# bisect是python内置模块，用于有序序列的插入和查找。


""" 三次样条插值 """
class Spline:
    """
    Cubic Spline class
    """
    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)    # dimension of x  有几个点就有几维
        h = np.diff(x)

        # calc coefficient a    计算系数a
        self.a = [iy for iy in y]

        # calc coefficient c    计算系数c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        # print(self.c1)

        # calc spline coefficient b and d   计算系数b和d
        for i in range(self.nx - 1):
            self.d.append( (self.c[i+1] - self.c[i]) / (3.0 * h[i]) )
            tb = (self.a[i+1] - self.a[i]) / h[i] - h[i] * (self.c[i+1] + 2.0*self.c[i]) / 3.0
            self.b.append(tb)


    def calc(self, t):
        """
        Calc position  计算每个点的位置
        if t is outside of the input x, return None
        """
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]          # 计算步长
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx**2.0 + self.d[i] * dx**3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative  求一次导数
        if t is outside of the input x, return None
        """
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative  求二次导数
        """
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        u"""
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c  计算A矩阵
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                       h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)
        return B


"""  """
class Spline2D:
    """
    2D Cubic Spline class
    """
    def __init__(self, x, y):
        self.s  = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)         # 求差
        dy = np.diff(y)         # 求差
        self.ds = [math.sqrt(idx**2 + idy**2) for (idx, idy) in zip(dx, dy)]    # 求两点之间的距离

        s = [0]
        s.extend(np.cumsum(self.ds))        # ？？？ 求和顺序
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)         # 计算每个输入s点对应的位置
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx  = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy  = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)

        k = (ddy*dx - ddx*dy) / (dx**2 + dy**2)

        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx  = self.sx.calcd(s)
        dy  = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)

        return yaw


def calc_spline_course(x, y, ds=0.1):
    sp = Spline2D(x, y)
    s  = list( np.arange(0, sp.s[-1], ds) )

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curveture(i_s))

    return rx, ry, ryaw, rk, s


