import numpy as np

from utils.ode_collection import ODE
from S3T2_solve_ode.py.one_step_methods import OneStepMethod


#  Коэффициенты методов Адамса
adams_coeffs = {
    1: [1],
    2: [-1 / 2, 3 / 2],
    3: [5 / 12, -4 / 3, 23 / 12],
    4: [-3 / 8, 37 / 24, -59 / 24, 55 / 24],
    5: [251 / 720, -637 / 360, 109 / 30, -1387 / 360, 1901 / 720]
}


def adams(ode: ODE, y_start, ts, coeffs, one_step_method: OneStepMethod):
    """
    Явный метод Адамса

    :param ode, y_start:    параметры задачи Коши
    :param ts:              список точек, по которым мы шагаем (шаг постоянный)
    :param coeffs:          список коэффициентов метода
    :param one_step_method: одношаговый метод для разгона
    :return:                список значений t (совпадает с ts), список значений y
    """
    h = np.abs((ts[len(ts) - 1] - ts[0])/(len(ts) - 1))#вычисляем постоянный шаг
    y = [y_start]
    for i in range(1, len(coeffs)):
            y.append(one_step_method.step(ode, ts[i - 1], y[i - 1], h))
    # предварительно вычисляем решения в len(coeffs)-1 точках через явный метод Эйлера
    f = [h * ode(ts[j], y[j]) for j in range(len(coeffs))]
    #вычисляем значения диф уравнения и умножояем на постоянный шаг len(coeffs)-1 раз
    for i in range(len(coeffs), len(ts)):
            y.append(y[i - 1] + np.dot(coeffs, f[i - len(coeffs):i]))
            f.append(h * ode(ts[i], y[i]))
    #вычисляем решения через явный метод Адамса
    return ts, y
    raise NotImplementedError
