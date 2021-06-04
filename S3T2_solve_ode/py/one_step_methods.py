import numpy as np
from copy import copy
from scipy.integrate import RK45, solve_ivp
from scipy.optimize import fsolve

import S3T2_solve_ode.py.coeffs_collection as collection
from utils.ode_collection import ODE


class OneStepMethod:
    def __init__(self, **kwargs):
        self.name = 'default_method'
        self.p = None  # порядок
        self.__dict__.update(**kwargs)

    def step(self, ode: ODE, t, y, dt):
        """
        делаем шаг: t => t+dt, используя ode(t, y)
        """
        return t + dt #делаем шаг
        raise NotImplementedError


class ExplicitEulerMethod(OneStepMethod):
    """
    Явный метод Эйлера (ничего менять не нужно)
    """
    def __init__(self):
        super().__init__(name='Euler (explicit)', p=1)

    def step(self, ode: ODE, t, y, dt):
        return y + dt * ode(t, y)


class ImplicitEulerMethod(OneStepMethod):
    """
    Неявный метод Эйлера
    Подробности: https://en.wikipedia.org/wiki/Backward_Euler_method
    """
    def __init__(self):
        super().__init__(name='Euler (implicit)', p=1)

    def step(self, ode: ODE, t, y, dt):
        y_next = lambda y_new: y + dt * ode(t + dt, y_new) - y_new
        return fsolve(y_next, y)
        raise NotImplementedError
#лямбда функция "анонимная" удобно использовать для быстрого напсиания кода.Грубо говоря, они ничем не отличается от обычного обьявления функции

class RungeKuttaMethod(OneStepMethod):
    """
    Явный метод Рунге-Кутты с коэффициентами (A, b)
    Замените метод step() так, чтобы он не использовал встроенный класс RK45
    """
    def __init__(self, coeffs: collection.RKScheme):
        super().__init__(**coeffs.__dict__)

    def step(self, ode: ODE, t, y, dt):
        A, b = self.A, self.b
        # rk = RK45(ode, t, y, t + dt)
        # rk.h_abs = dt
        # rk.step()
        # return rk.y

        K = np.zeros((np.size(b), len(y)))#Ki
        c = np.sum(A, axis=1)#axis=1- построчное направление в матрице
        for i in range(0, np.size(b)):
            if i == 0:
                K[i] = np.array(dt * ode(t + dt, y))
            else:
                K[i] = np.array(dt * ode(t + c[i] * dt, y + np.dot(A[i], K)))
        """
        Так как в данном случае мы работаем с матрицами (где количество измирений 1 (dim), для данного теста), то для Ki
        создаем матрицу (i,j), из-за того что в тесте матрица y (1,2), а b (1,4), то i=4 (i=np.size(b), j=2 (j=len(y)). 
        """
        return y + np.dot(b, K)#находим приближение



class EmbeddedRungeKuttaMethod(RungeKuttaMethod):
    """
    Вложенная схема Рунге-Кутты с параметрами (A, b, e):
    """
    def __init__(self, coeffs: collection.EmbeddedRKScheme):
        super().__init__(coeffs=coeffs)

    def embedded_step(self, ode: ODE, t, y, dt):
        """
        Шаг с использованием вложенных методов:
        y1 = RK(ode, A, b)
        y2 = RK(ode, A, b+e)

        :return: приближение на шаге (y1), разность двух приближений (dy = y2-y1)
        """
        A, b, e = self.A, self.b, self.e
        c = np.sum(A, axis=1)
        K = np.zeros((np.size(b), len(y)))
        for i in range(0, np.size(b)):
            if i == 0:
                K[i] = np.array(dt * ode(t + dt, y))
            else:
                K[i] = np.array(dt * ode(t + c[i] * dt, y + np.dot(A[i], K)))
        return y + np.dot(b, K), np.dot(e, K)



class EmbeddedRosenbrockMethod(OneStepMethod):
    """
    Вложенный метод Розенброка с параметрами (A, G, gamma, b, e)
    Подробности: https://dl.acm.org/doi/10.1145/355993.355994 (уравнение 2)
    """
    def __init__(self, coeffs: collection.EmbeddedRosenbrockScheme):
        super().__init__(**coeffs.__dict__)

    def embedded_step(self, ode: ODE, t, y, dt):
        """
        Шаг с использованием вложенных методов:
        y1 = Rosenbrock(ode, A, G, gamma, b)
        y2 = Rosenbrock(ode, A, G, gamma, b+e)

        :return: приближение на шаге (y1), разность двух приближений (dy = y2-y1)
        """
        A, G, g, b, e = self.A, self.G, self.gamma, self.b, self.e
        c = np.sum(A, axis=1)
        K = np.zeros((np.size(b), len(y)))
        E = np.linalg.inv(np.eye(len(y)) - dt * ode.jacobian(t, y) * g)
        for j in range(np.size(b)):
            K[j] = E.dot(dt * ode(t + c[j] * dt, y + np.dot(A[j], K)) +
                         dt * ode.jacobian(t, y).dot(np.dot(G[j], K)))
        return y + b.dot(K), e.dot(K)
        return y1, dy
