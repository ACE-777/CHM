import enum
import numpy as np

from utils.ode_collection import ODE
from S3T2_solve_ode.py.one_step_methods import OneStepMethod


class AdaptType(enum.Enum):
    RUNGE = 0
    EMBEDDED = 1


def fix_step_integration(method: OneStepMethod, ode: ODE, y_start, ts):
    """
    Интегрирование одношаговым методом с фиксированным шагом

    :param method:  одношаговый метод
    :param ode:     СОДУ
    :param y_start: начальное значение
    :param ts:      набор значений t
    :return:        список значений t (совпадает с ts), список значений y
    """
    ys = [y_start]

    for i, t in enumerate(ts[:-1]):
        y = ys[-1]

        y1 = method.step(ode, t, y, ts[i + 1] - t)
        ys.append(y1)

    return ts, ys


def adaptive_step_integration(method: OneStepMethod, ode: ODE, y_start, t_span,
                              adapt_type: AdaptType,
                              atol, rtol):
    """
    Интегрирование одношаговым методом с адаптивным выбором шага.
    Допуски контролируют локальную погрешность:
        err <= atol
        err <= |y| * rtol

    :param method:      одношаговый метод
    :param ode:         СОДУ
    :param y_start:     начальное значение
    :param t_span:      интервал интегрирования (t0, t1)
    :param adapt_type:  правило Рунге (AdaptType.RUNGE) или вложенная схема (AdaptType.EMBEDDED)
    :param atol:        допуск на абсолютную погрешность
    :param rtol:        допуск на относительную погрешность
    :return:            список значений t (совпадает с ts), список значений y
    """
    y = y_start
    t, t_end = t_span
    p = method.p + 1#порядок точности
    ys = [y]
    ts = [t]

    delta = (1/max(np.abs(t), np.abs(t_end)))**(p + 1) + np.linalg.norm(ode(t, y))**(p + 1)
    h1 = (atol/delta)**(1/(p + 1))
    u1 = y + h1 * ode(t, y)#делаем один шаг методом Эйлера, получаем приближение u1 в точке t+h1
    delta = (1/max(np.abs(t), np.abs(t_end)))**(p + 1) + np.linalg.norm(ode(t + h1, u1))**(p + 1)
    h2 = (atol/delta)**(1/(p + 1))#получаем приближение для шага h2, взяв точку (t+h1,u1)
    # выбираем начальный шаг
    h_opt = min(h1, h2)

    #адаптивный выбор шага
    while t < t_end:
        if t + h_opt > t_end:
            h_opt = t_end - t
        if adapt_type == AdaptType.RUNGE:
            y1 = method.step(ode, t, y, h_opt)
            y2 = method.step(ode, t + h_opt/2, method.step(ode, t, y, h_opt/2), h_opt/2)
            err = (y2 - y1)/(1-2**(-p))
            y_h = y1 + err
        else:
            y_h, err = method.embedded_step(ode, t, y, h_opt)
        rtol = atol
        if np.linalg.norm(err) <= rtol:
            y = y_h
            ts.append(t + h_opt) #добовляем меньший шаг
            t = t + h_opt
            ys.append(y)
        h_opt = h_opt * (rtol/np.linalg.norm(err))**(1/(p+1))

    return ts, ys
