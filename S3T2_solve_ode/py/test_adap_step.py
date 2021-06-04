import pytest
import numpy as np
import matplotlib.pyplot as plt

from utils.ode_collection import Harmonic, HarmExp, Arenstorf
from utils.utils import get_accuracy
from S3T2_solve_ode.py.solve_ode import adaptive_step_integration, AdaptType
from S3T2_solve_ode.py.one_step_methods import (
    ExplicitEulerMethod,
    RungeKuttaMethod,
    EmbeddedRungeKuttaMethod,
)
import S3T2_solve_ode.py.coeffs_collection as coeffs


@pytest.mark.parametrize('ode,y0', (
        (
                Harmonic(np.array([1., 1.]), 1, 1),
                np.array([1., 1.])
        ),
        (
                HarmExp(),
                np.exp([1, 0])
        ),
))
def test_adaptive(ode, y0):
    """
    Проверяем алгоритмы выбора шага
    """
    t0, t1 = 0, 4*np.pi

    atol = 1e-6
    rtol = 1e-3

    tss = []
    yss = []

    methods = (
        (ExplicitEulerMethod(),                         AdaptType.RUNGE),
        (RungeKuttaMethod(coeffs.rk4_coeffs),           AdaptType.RUNGE),
        (EmbeddedRungeKuttaMethod(coeffs.dopri_coeffs), AdaptType.EMBEDDED),
    )

    for method, adapt_type in methods:
        ode.clear_call_counter()
        ts, ys = adaptive_step_integration(method=method,
                                           ode=ode, y_start=y0, t_span=(t0, t1),
                                           adapt_type=adapt_type,
                                           atol=atol, rtol=rtol)
        print(f'{method.name} took {ode.get_call_counter()} function calls')

        tss.append(np.array(ts))
        yss.append(ys)

    ts = np.array(sorted([t for ts in tss for t in ts]))
    exact = ode[ts].T
    y0 = np.array([y[0] for y in exact])

    # plots
    fig1, ax1 = plt.subplots(num='y(t)')
    fig1.suptitle('test_adaptive: y(t)')
    ax1.set_xlabel('t'), ax1.set_ylabel('y')
    ax1.plot(ts, y0, 'ko-', label='exact')

    fig2, ax2 = plt.subplots(num='dt(t)')
    fig2.suptitle('test_adaptive: step sizes')
    ax2.set_xlabel('t'), ax2.set_ylabel('dt')

    fig3, ax3 = plt.subplots(num='dy(t)')
    fig3.suptitle('test_adaptive: accuracies')
    ax3.set_xlabel('t'), ax3.set_ylabel('accuracy')

    for (m, _), ts, ys in zip(methods, tss, yss):
        ax1.plot(ts, [y[0] for y in ys], '.', label=m.name)
        ax2.plot(ts[:-1], ts[1:] - ts[:-1], '.-', label=m.name)
        ax3.plot(ts, get_accuracy(ode[ts].T, ys), '.-', label=m.name)

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()


def test_adaptive_order():
    """
    Проверяем сходимость
    Q: почему наклон линии (число в скобках) соответствует порядку метода?
    """
    t0, t1 = 0, 2*np.pi
    y0 = np.array([1., 1.])
    ode = Harmonic(y0, 1, 1)

    methods = (
        (ExplicitEulerMethod(),                         AdaptType.RUNGE),
        (RungeKuttaMethod(coeffs.rk4_coeffs),           AdaptType.RUNGE),
        (RungeKuttaMethod(coeffs.dopri_coeffs),         AdaptType.RUNGE),
        (EmbeddedRungeKuttaMethod(coeffs.dopri_coeffs), AdaptType.EMBEDDED),
    )
    tols = 10. ** -np.arange(3, 9)

    plt.figure(figsize=(9, 6))
    for i, (method, adapt_type) in enumerate(methods):
        print(method.name)
        fcs = []
        errs = []
        for tol in tols:
            ode.clear_call_counter()
            ts, ys = adaptive_step_integration(method=method,
                                               ode=ode,
                                               y_start=y0,
                                               t_span=(t0, t1),
                                               adapt_type=adapt_type,
                                               atol=tol, rtol=tol*1e3)
            err = np.linalg.norm(ys[-1] - ode[t1])
            fc = ode.get_call_counter()
            print(f'{method.name} with {adapt_type.name}: {fc} RHS calls, err = {err:.5f}')
            errs.append(err)
            fcs.append(fc)

        x = np.log10(fcs)
        y = -np.log10(errs)
        k, b = np.polyfit(x, y, 1)
        plt.plot(x, k*x+b, 'k:')
        plt.plot(x, y, 'p', label=f'{method.name} {adapt_type} ({k:.2f})')
    plt.suptitle('test_adaptive_order: check RHS evals')
    plt.xlabel('log10(function_calls)')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def test_arenstorf():
    """
    https://en.wikipedia.org/wiki/Richard_Arenstorf#The_Arenstorf_Orbit
    https://commons.wikimedia.org/wiki/File:Arenstorf_Orbit.gif
    Q: какие участки траектории наиболее быстрые?
    """
    ode = Arenstorf()
    t0, t1 = 0, 1 * ode.t_period
    y0 = ode.y0

    atol = 1e-6
    rtol = 1e-3

    tss = []
    yss = []

    methods = (
        # (ExplicitEulerMethod(),                         AdaptType.RUNGE),
        (RungeKuttaMethod(coeffs.rk4_coeffs),           AdaptType.RUNGE),
        (EmbeddedRungeKuttaMethod(coeffs.dopri_coeffs), AdaptType.EMBEDDED),
    )

    fig1, ax1 = plt.subplots(num='traj')
    fig1.suptitle('Arenstorf orbit: trajectory')
    ax1.set_xlabel('x1'), ax1.set_ylabel('x2')

    fig2, ax2 = plt.subplots(num='dt(t)')
    fig2.suptitle('Arenstorf orbit: step sizes')
    ax2.set_xlabel('t'), ax2.set_ylabel('dt')

    fig3, ax3 = plt.subplots(num='|f|')
    fig3.suptitle('Arenstorf orbit: RHS analysis')
    ax3.set_xlabel('t')

    for method, adapt_type in methods:
        ts, ys = adaptive_step_integration(method=method,
                                           ode=ode,
                                           y_start=y0,
                                           t_span=(t0, t1),
                                           adapt_type=adapt_type,
                                           atol=atol, rtol=rtol)
        tss.append(np.array(ts))
        yss.append(ys)

    for (m, _), ts, ys in zip(methods, tss, yss):
        ax1.plot([y[0] for y in ys],
                 [y[1] for y in ys],
                 ':', label=m.name)
        ax2.plot(ts[:-1], ts[1:] - ts[:-1], '.-', label=m.name)

    derivatives = [np.linalg.norm(ode(t, y)) for t, y in zip(ts, ys)]
    # derivatives = [1/(np.linalg.norm(ode(t, y))) for t, y in zip(ts, ys)]

    ax1.plot(0, 0, 'bo', label='Earth')
    ax1.plot(1, 0, '.', color='grey', label='Moon')
    ax3.plot(ts, derivatives, label='|f(t, y)|')
    ax1.legend()
    ax2.legend()
    plt.show()
