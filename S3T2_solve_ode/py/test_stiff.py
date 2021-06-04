import numpy as np
import matplotlib.pyplot as plt

from utils.ode_collection import VanDerPol
from S3T2_solve_ode.py.solve_ode import adaptive_step_integration, AdaptType
from S3T2_solve_ode.py.one_step_methods import (
    ExplicitEulerMethod,
    ImplicitEulerMethod,
    EmbeddedRosenbrockMethod,
)
import S3T2_solve_ode.py.coeffs_collection as coeffs


def test_stiff():
    """
    Проверяем явные и неявные методы на жёсткой задаче
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    Q: почему даже метод Розенброка иногда уменьшает шаг почти до нуля?
    """
    t0 = 0
    t1 = 800*np.pi

    mu = 1000
    y0 = np.array([2., 0.])
    ode = VanDerPol(y0, mu)

    fig1, ax1 = plt.subplots()
    fig2, (ax21, ax22) = plt.subplots(1, 2)
    fig3, ax3 = plt.subplots()

    colors = 'rgbcmyk'
    for i, (method, adapt_type) in enumerate(
            [
                (ExplicitEulerMethod(),                                AdaptType.RUNGE),
                (ImplicitEulerMethod(),                                AdaptType.RUNGE),
                (EmbeddedRosenbrockMethod(coeffs.rosenbrock23_coeffs), AdaptType.EMBEDDED),
            ]
    ):
        ode.clear_call_counter()
        ts, ys = adaptive_step_integration(method, ode, y0, (t0, t1),
                                           adapt_type=adapt_type,
                                           atol=1e-6, rtol=1e-3)
        print(f'{method.name}: {len(ts)-1} steps, {ode.get_call_counter()} RHS calls')

        ax1.plot([y[0] for y in ys],
                 [y[1] for y in ys],
                 f'{colors[i]}.--', label=method.name)

        ax21.plot(ts,
                  [y[0] for y in ys],
                  f'{colors[i]}.--', label=method.name)
        ax22.plot(ts,
                  [y[1] for y in ys],
                  f'{colors[i]}.--', label=method.name)

        ax3.plot(ts[:-1],
                 np.array(ts[1:]) - np.array(ts[:-1]),
                 f'{colors[i]}.--', label=method.name)

    ax1.set_xlabel('x'), ax1.set_ylabel('y'), ax1.legend()
    fig1.suptitle(f'test_stiff: Van der Pol, mu={mu:.2f}, y(x)')

    ax21.set_xlabel('t'), ax21.set_ylabel('x'), ax21.legend()
    ax22.set_xlabel('t'), ax22.set_ylabel('y'), ax22.legend()
    fig2.suptitle(f'test_stiff: Van der Pol, mu={mu:.2f}, x(t)')

    ax3.set_xlabel('t'), ax3.set_ylabel('dt'), ax3.legend()
    fig3.suptitle(f'test_stiff: Van der Pol, mu={mu:.2f}, dt(t)')

    plt.show()
