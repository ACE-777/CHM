import numpy as np


def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0):
    """
    Вычисляем моменты весовой функции с 0-го по max_s-ый на интервале [xl, xr]
    Весовая функция: p(x) = 1 / (x-a)^alpha / (b-x)^beta, причём гарантируется, что:
        1) 0 <= alpha < 1
        2) 0 <= beta < 1
        3) alpha * beta = 0

    :param max_s:   номер последнего момента
    :return:        список значений моментов
    """
    assert alpha * beta == 0, f'alpha ({alpha}) and/or beta ({beta}) must be 0'

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]

    if alpha != 0.0:
        assert a is not None, f'"a" not specified while alpha != 0'

        array = []
        for s in range(0, max_s + 1):
            sum = 0
            for i in range(1, s + 2):
                to_add = 0
                to_add += (xr - a) ** (-alpha + i) * xr ** (s + 1 - i) * (-1) ** (i + 1)
                to_add -= (xl - a) ** (-alpha + i) * xl ** (s + 1 - i) * (-1) ** (i + 1)

                for k in range(1, i + 1):
                    to_add /= (-alpha + k)

                for k in range(0, i - 1):
                    to_add *= s - k

                sum += to_add
            array.append(sum)

        return array
    if beta != 0.0:
        assert b is not None, f'"b" not specified while beta != 0'

        array = []
        for s in range(0, max_s + 1):
            sum = 0
            for i in range(1, s + 2):
                to_add = 0
                to_add -= (b - xr) ** (-beta + i) * xr ** (s + 1 - i)
                to_add += (b - xl) ** (-beta + i) * xl ** (s + 1 - i)
                for k in range(1, i + 1):
                    to_add /= (-beta + k)

                for k in range(0, i - 1):
                    to_add *= s - k
                sum += to_add

            array.append(sum)

        return array

    m = np.zeros(max_s+1)
    return m
    raise NotImplementedError


def runge(s0, s1, m, L):
    """
    Оценка погрешности последовательных приближений s0 и s1 по правилу Рунге

    :param m:   порядок погрешности
    :param L:   кратность шага
    :return:    оценки погрешностей s0 и s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    Оценка порядка главного члена погрешности по последовательным приближениям s0, s1 и s2 по правилу Эйткена
    Считаем, что погрешность равна R(h) = C*h^m + o(h^m)

    :param L:   кратность шага
    :return:    оценка порядка главного члена погрешности (m)
    """
    if s2 == s1 or s1 == s0 or ((s2-s1)/(s1-s0) < 0):
        return -1
    else:
        return -np.log((s2-s1)/(s1-s0))/np.log(L)
    raise NotImplementedError

    """
    В 89 строке мы возврощаем "-1", так как значение квадратурных сумм могут совподать или же подлогарифмическое 
    выражение может быть отрицательным (квадратурные суммы могут находится с разных сторон от точной велечины интеграла).
    В данном случае это приводит к грубой оценки параметра m. 
    """


def quad(func, x0, x1, xs, **kwargs):
    """
    Интерполяционная квадратурная формула

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param xs:      узлы
    :param kwargs:  параметры весовой функции (должны передаваться в moments)
    :return:        значение ИКФ
    """
    m = moments(len(xs) - 1, x0, x1, **kwargs)

    X = []

    for i in range(len(m)):
        v = []
        for s in xs:
            v.append(s ** i)
        X.append(v)

    A = np.linalg.solve(X, m)

    Fx = []

    for s in xs:
        Fx.append(func(s))

    result = np.dot(A, Fx)
    return result

    raise NotImplementedError


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    Интерполяционная квадратурная формула типа Гаусса

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param n:       количество узлов
    :param kwargs:  параметры весовой функции (должны передаваться в moments)
    :return:        значение ИКФ
    """

    m = np.array(moments(2*n - 1, x0, x1, **kwargs))

    C = []
    B = []

    for s in range(n):
        v = []
        for j in range(n):
            v.append(m[j+s])
        C.append(v)
        B.append(-m[n+s])

    if np.linalg.det(C) == 0:
        return 0

    A = np.flip(np.linalg.solve(C, B)) #меняем порядок коэффициентов для дальнейшего вычисления узлового многочлена
    A = np.insert(A, 0, 1)#добовляем первый недостоющий коэф (по алгоритму равный 1) для узлового многочлена
    X = np.roots(A)
    return quad(func, x0, x1, X, **kwargs)


    raise NotImplementedError


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    Составная квадратурная формула

    :param func:        интегрируемая функция
    :param x0, x1:      интервал
    :param n_intervals: количество интервалов
    :param n_nodes:     количество узлов на каждом интервале
    :param kwargs:      параметры весовой функции (должны передаваться в moments)
    :return:            значение СКФ
    """
    mesh = np.linspace(x0, x1, n_intervals + 1)
    return sum(quad(func, mesh[i], mesh[i+1], interval(n_nodes, mesh[i], mesh[i+1]), **kwargs) for i in range(n_intervals))
    raise NotImplementedError

def interval(n, xl, xr):
    if n == 1:
        return[0.5*(xl+xr)]#в случае единсвтенной узловой точки, устанавливаем ее на середину
    else:
        return np.linspace(xl, xr, n)#делаем равностоящие узлы

def integrate(func, x0, x1, tol):
    """
    Интегрирование с заданной точностью (error <= tol)

    Оцениваем сходимость по Эйткену, потом оцениваем погрешность по Рунге и выбираем оптимальный размер шага
    Делаем так, пока оценка погрешности не уложится в tol

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param tol:     допуск
    :return:        значение интеграла, оценка погрешности
    """
    N = 1 #intervals
    n_nodes = 2 #nodes
    L = 2

    S_0 = composite_quad(func, x0, x1, N, n_nodes)
    N *= L
    S_1 = composite_quad(func, x0, x1, N, n_nodes)
    N *= L
    S_2 = composite_quad(func, x0, x1, N, n_nodes)

    m = aitken(S_0, S_1, S_2, L)

    d0, d1 = runge(S_0, S_1, m, L)

    while d1 > tol:#изменяем размерности сеток до тех пор, пока не добьемся заданной точности вычисления(tol)
        S_0 = S_1
        S_1 = S_2
        N *= L
        S_2 = composite_quad(func, x0, x1, N, n_nodes)
        m = aitken(S_0, S_1, S_2, L)
        d0, d1 = runge(S_0, S_1, m, L)
    
    return S_1, d1
    raise NotImplementedError
