import pytest
import numpy as np
import matplotlib.pyplot as plt

from utils.ode_collection import Harmonic
from utils.utils import get_accuracy
from S3T2_solve_ode.py.solve_ode import fix_step_integration
from S3T2_solve_ode.py.one_step_methods import (
    ExplicitEulerMethod,
    ImplicitEulerMethod,
    RungeKuttaMethod,
)
import S3T2_solve_ode.py.coeffs_collection as collection
from S3T2_solve_ode.py.multistep_methods import adams, adams_coeffs


def test_one_step():
    """
    Проверяем методы Эйлера и Рунге-Кутты
    """
    y0 = np.array([0., 1.])
    t0 = 0
    t1 = np.pi/2
    dt = 0.1

    ode = Harmonic(y0, 1, 1)
    ts = np.arange(t0, t1+dt, dt)

    exact = ode[ts].T
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(ts, [e[0] for e in exact], 'k', label='Exact')

    colors = 'rgbcmyk'
    for i, method in enumerate(
            [
                ExplicitEulerMethod(),
                ImplicitEulerMethod(),
                RungeKuttaMethod(collection.rk4_coeffs),
                RungeKuttaMethod(collection.dopri_coeffs),
            ]
    ):
        ode.clear_call_counter()
        _, y = fix_step_integration(method, ode, y0, ts)
        n_calls = ode.get_call_counter()
        print(f'One-step {method.name}: {len(y)-1} steps, {n_calls} function calls')

        ax1.plot(ts,
                 [_y[0] for _y in y],
                 f'{colors[i]}.--', label=method.name)
        ax2.plot(ts,
                 get_accuracy(exact, y),
                 f'{colors[i]}.--', label=method.name)

    ax1.set_xlabel('t'), ax1.set_ylabel('y'), ax1.legend()
    ax2.set_xlabel('t'), ax2.set_ylabel('accuracy'), ax2.legend()
    plt.suptitle('test_one_step')
    plt.show()


def test_multi_step():
    """
    Проверяем методы Адамса
    Q: сравните правые графики для обоих случаев и объясните разницу
    """
    y0 = np.array([0., 1.])
    t0 = 0
    t1 = 1.
    dt = 0.1

    f = Harmonic(y0, 1, 1)
    ts = np.arange(t0, t1+dt, dt)
    exact = f[ts].T

    for one_step_method in [
        RungeKuttaMethod(collection.rk4_coeffs),
        ExplicitEulerMethod(),
    ]:
        _, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(ts, [e[0] for e in exact], 'k', label='Exact')
        for p, c in adams_coeffs.items():
            f.clear_call_counter()
            t_adams, y_adams = adams(f, y0, ts, c,
                                     one_step_method=one_step_method)
            n_calls = f.get_call_counter()
            print(f'{p}-order multi-step with one-step {one_step_method.name}: {n_calls} function calls')

            err = get_accuracy(exact, y_adams)

            label = f"Adams's order {p}"
            ax1.plot(t_adams, [y[0] for y in y_adams], '.--', label=label)
            ax2.plot(t_adams, err, '.--', label=label)

        ax1.set_xlabel('t'), ax1.set_ylabel('y'), ax1.legend()
        ax2.set_xlabel('t'), ax2.set_ylabel('accuracy'), ax2.legend()
        plt.suptitle(f'test_multi_step\none step method: {one_step_method.name}')
    plt.show()
