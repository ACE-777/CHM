import pytest
import numpy as np
from scipy.integrate import quad as sp_quad
import matplotlib.pyplot as plt

from utils.integrate_collection import Monome, Harmonic

from utils.utils import get_accuracy
from S3T1_integration.py.integration import (quad,
                                             quad_gauss,
                                             composite_quad,
                                             integrate,
                                             aitken, interval)


@pytest.mark.parametrize('n_nodes', [2, 3, 5])
def test_composite_quad(n_nodes):
    """
    Проверяем 2-, 3-, 5-узловые СКФ
    Q: объясните скорость сходимости для каждого случая
    """
    fig, ax = plt.subplots(1, 2)

    x0, x1 = 0, 1
    L = 2
    n_intervals = [L ** q for q in range(0, 8)]

    for i, degree in enumerate((5, 6)):
        p = Monome(degree)
        Y = [composite_quad(p, x0, x1, n_intervals=n, n_nodes=n_nodes) for n in n_intervals]
        accuracy = get_accuracy(Y, p[x0, x1] * np.ones_like(Y))
        x = np.log10(n_intervals)

        # оценка сходимости
        ind = np.isfinite(x) & np.isfinite(accuracy)
        k, b = np.polyfit(x[ind], accuracy[ind], 1)
        aitken_degree = aitken(*Y[0:6:2], L ** 2)

        ax[i].plot(x, k*x+b, 'b:', label=f'{k:.2f}*x+{b:.2f}')
        ax[i].plot(x, aitken_degree*x+b, 'm:', label=f'aitken ({aitken_degree:.2f})')
        ax[i].plot(x, accuracy, 'kh', label=f'accuracy for x^{degree}')
        ax[i].set_title(f'{n_nodes}-node CQ for x^{degree}')
        ax[i].set_xlabel('log10(n_intervals)')
        ax[i].set_ylabel('accuracy')
        ax[i].legend()

        if n_nodes < degree:
            assert np.abs(aitken_degree - k) < 0.5, \
                f'Aitken estimation {aitken_degree:.2f} is too far from actual {k:.2f}'

    plt.show()


@pytest.mark.parametrize('v', [2, 3, 5, 6])
def test_composite_quad_degree(v):
    """
    Проверяем сходимость СКФ при наличии неравных весов
    Q: скорость сходимости оказывается дробной, почему?
    """
    from .variants import params
    a, b, alpha, beta, f = params(v)
    x0, x1 = a, b
    # a, b = -10, 10

    L = 2
    n_intervals = [L ** q for q in range(2, 11)]
    n_nodes = 3

    exact = sp_quad(lambda x: f(x) / (x-a)**alpha / (b-x)**beta, x0, x1)[0]

    Y = [composite_quad(f, x0, x1, n_intervals=n, n_nodes=n_nodes,
                        a=a, b=b, alpha=alpha, beta=beta) for n in n_intervals]
    accuracy = get_accuracy(Y, exact * np.ones_like(Y))

    x = np.log10(n_intervals)
    aitken_degree = aitken(*Y[5:8], L)
    a1, a0 = np.polyfit(x, accuracy, 1)
    assert a1 > 1, 'composite quad did not converge!'

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # график весовой функции
    xs = np.linspace(x0, x1, n_intervals[-1]+1)
    ys = 1 / ((xs-a)**alpha * (b-xs)**beta)

    ax1.plot(xs, ys, label='weights')
    ax = list(ax1.axis())
    ax[2] = 0.
    ax1.axis(ax)
    ax1.set_xlabel('x')
    ax1.set_ylabel('p(x)')
    ax1.legend()

    # график точности
    ax2.plot(x, accuracy, 'kh')
    ax2.plot(x, a1*x+a0, 'b:', label=f'{a1:.2f}*x+{a0:.2f}')
    ax2.set_xlabel('log10(n_intervals)')
    ax2.set_ylabel('accuracy')
    ax2.legend()

    fig.suptitle(f'variant #{v} (alpha={alpha:4.2f}, beta={beta:4.2f})\n'
                 f'aitken estimation: {aitken_degree:.2f}')
    plt.show()


def test_quad_vs_cq():
    """
    Проверяем, как работают ИКФ и СКФ при одинакомом количестве обращений к функции
    """
    x0, x1 = 0, np.pi/2

    p = Harmonic(0, 1)
    y0 = p[x0, x1]

    n_nodes = 2
    ys_nt_ct = []
    ys_gauss = []
    ys_cquad = []

    n_intervals = 2 ** np.arange(8)
    for n in n_intervals:
        n_evals = n * n_nodes
        ys_nt_ct.append(quad(p, x0, x1, np.linspace(0, 1, n_evals) * (x1-x0) + x0))
        ys_gauss.append(quad_gauss(p, x0, x1, n_evals))
        ys_cquad.append(composite_quad(p, x0, x1, n, n_nodes))

    for ys, label in zip((ys_nt_ct, ys_gauss, ys_cquad),
                         (f'newton-cotes', f'gauss', f'{n_nodes}-node CQ')):
        acc = get_accuracy(ys, y0 * np.ones_like(ys))
        acc[acc > 17] = 17

        plt.plot(np.log10(n_intervals), acc, '.:', label=label)

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('log10(n_evals)')
    plt.suptitle(f'test quad vs composite quad')
    plt.show()


def test_integrate():
    """
    Интегрируем с заданной точностью
    """
    p = Harmonic(1, 0)
    x0, x1 = 0, np.pi

    for tol in 10. ** np.arange(-9, -2):
        s, err = integrate(p, x0, x1, tol=tol)

        print(f'Check for tol {tol:.2e}: res = {s-err:.6f} .. {s:.6f} .. {s+err:.6f}')

        assert err >= 0,                            'estimated error should be >= 0'
        assert np.abs(p[x0, x1] - s) <= 1.1*err,    'actual error should be <= estimated error + 10%'
        assert np.abs(p[x0, x1] - s) <= tol,        'actual error should be <= tolerance'
