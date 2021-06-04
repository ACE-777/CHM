import numpy as np
import scipy


class RKScheme:
    def __init__(self, name, A, b, p):
        """
        Схема Рунге-Кутты

        :param A, b:    параметры схемы
        :param p:       порядок
        """
        self.name = name
        self.A = np.array(A)
        self.b = np.array(b)
        self.p = p


class EmbeddedRKScheme(RKScheme):
    def __init__(self, name, A, b, e, p, q):
        """
        Вложенная схема Рунге-Кутты

        :param A, b, e: параметры схемы
        :param p:       порядок схемы (A, b)
        :param q:       порядок схемы (A, b+e)
        """
        super().__init__(name, A, b, p)
        self.e = np.array(e)
        self.q = q


class EmbeddedRosenbrockScheme:
    def __init__(self, name, A, G, gamma, b, e, p, q):
        """
        Вложенная схема Розенброка

        :param A, G, gamma, b, e:   параметры схемы
        :param p:                   порядок схемы (A, G, gamma, b)
        :param q:                   порядок схемы (A, G, gamma, b+e)
        """
        self.name = name
        self.A = np.array(A)
        self.G = np.array(G)
        self.b = np.array(b)
        self.gamma = gamma
        self.p = p
        self.e = np.array(e)
        self.q = q


#  Классический метод Рунге-Кутты 4 порядка
rk4_coeffs = RKScheme(
    name='RK4',
    A=[
        [0.0, 0.0, 0.0, 0.],
        [0.5, 0.0, 0.0, 0.],
        [0.0, 0.5, 0.0, 0.],
        [0.0, 0.0, 1.0, 0.],
    ],
    b=np.array([1, 2, 2, 1]) / 6,
    p=4,
)


#  Метод Дорманда-Принса 5(4)
dopri_coeffs = EmbeddedRKScheme(
    name='DoPri5(4)',
    A=[
        [0,          0,           0,          0,        0,           0,     0],
        [1/5,        0,           0,          0,        0,           0,     0],
        [3/40,       9/40,        0,          0,        0,           0,     0],
        [44/45,      -56/15,      32/9,       0,        0,           0,     0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0,           0,     0],
        [9017/3168,  -355/33,     46732/5247, 49/176,   -5103/18656, 0,     0],
        [35/384,     0,           500/1113,   125/192,  -2187/6784,  11/84, 0],
    ],
    b=[35/384,    0, 500/1113, 125/192,  -2187/6784,   11/84,   0],
    e=[-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40],
    p=5,
    q=4,
)

#  Метод Розенброка 2(3)
v2 = np.sqrt(2)
d = 1/(2+v2)
rosenbrock23_coeffs = EmbeddedRosenbrockScheme(
    name='Rosenbrock2(3)',
    A=[
        [0,   0, 0],
        [1/2, 0, 0],
        [0,   1, 0],
    ],
    G=[
        [0.,   0,      0],
        [-d,   0,      0],
        [3-v2, 2*v2-5, 0],
    ],
    gamma=d,
    b=[0,   1, 0],
    e=[1/6, -1/3, 1/6],
    p=2,
    q=3,
)
