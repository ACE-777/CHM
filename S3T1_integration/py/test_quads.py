import pytest
import numpy as np
import matplotlib.pyplot as plt

from utils.integrate_collection import Monome, Harmonic

from utils.utils import get_accuracy
from S3T1_integration.py.integration import (quad,
                                             quad_gauss,
                                             moments)


@pytest.mark.parametrize('func', [
    np.sin,
    np.cos,
    np.exp,
    np.sqrt
])
def test_interpolation(func):
    """
    Проверяем, что значит буква "И" в названии ИКФ
    Интегрируем интерполяционный многочлен, затем сравниваем результат с quad()
    """
    x0, x1 = 0, 1
    n_nodes = 5

    xs = np.linspace(x0, x1, n_nodes)
    ys = func(xs)

    # numpy
    poly = np.polyfit(xs, ys, deg=n_nodes-1)
    polyint = np.polyint(poly)
    int_v = np.polyval(polyint, x1) - np.polyval(polyint, x0)

    # our
    num_v = quad(func, x0, x1, xs)

    delta = np.abs(int_v - num_v)
    assert delta < 1e-6

    print(f'interpolate+integrate: {int_v:8.3f}, quad: {num_v:8.3f}, delta: {delta:e}')


def test_quad_degree():
    """
    Проверяем АСТ для ИКФ
    Q: почему в некоторых случаях x^n интегрируется почти без ошибок при n узлах ИКФ?
    """
    x0, x1 = 0, 1

    max_degree = 9
    max_nodes = 7

    for deg in range(max_degree):
        p = Monome(deg)
        y0 = p[x0, x1]

        node_counts = range(1, max_nodes+1)

        Y = [quad(p, x0, x1, np.linspace(x0, x1, node_count)) for node_count in node_counts]
        # Y = [quad(p, x0, x1, x0 + (x1-x0) * np.random.random(node_count)) for node_count in node_counts]
        accuracy = get_accuracy(Y, y0 * np.ones_like(Y))

        # Проверяем точность
        for node_count, acc in zip(node_counts, accuracy):
            if node_count >= deg + 1:
                assert acc > 6

        plt.plot(node_counts, accuracy, '.:', label=f'x^{deg}')

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('node_count')
    plt.suptitle(f'test quad')
    plt.show()


def test_weighted_quad_degree():
    """
    Проверяем АСТ для ИКФ с весами
    Посчитаем n-ый момент весовой функции двумя способами:
        - через moments()
        - численно через quad()
    """
    x0, x1 = 1, 3
    alpha = 0.14
    beta = 0.88

    max_degree = 7
    for deg in range(1, max_degree):
        p = Monome(deg)
        xs = np.linspace(x0, x1, 6)[1:-1]  # 4 points => accuracy degree is 3

        res = quad(p, x0, x1, xs, a=x0, alpha=alpha)
        ans = moments(deg, x0, x1, a=x0, alpha=alpha)[-1]
        d = abs(res - ans)
        print(f'{deg:2}-a: {res:8.3f} vs {ans:8.3f}, delta = {d:e}')
        if deg < len(xs):
            assert d < 1e-6

        res = quad(p, x0, x1, xs, b=x1, beta=beta)
        ans = moments(deg, x0, x1, b=x1, beta=beta)[-1]
        d = abs(res - ans)
        print(f'{deg:2}-b: {res:8.3f} vs {ans:8.3f}, delta = {d:e}')
        if deg < len(xs):
            assert d < 1e-6


def test_quad_gauss_degree():
    """
    Проверяем АСТ для ИКФ Гаусса
    """
    x0, x1 = 0, 1

    max_degree = 9 #степень мнг

    for deg in range(max_degree):
        p = Monome(deg)
        y0 = p[x0, x1]

        node_counts = range(1, 7)  # кол узлов
        Y = [quad_gauss(p, x0, x1, node_count) for node_count in node_counts]
        accuracy = get_accuracy(Y, y0 * np.ones_like(Y))

        # Проверяем точность
        for node_count, acc in zip(node_counts, accuracy):
            if 2 * node_count >= deg + 1:
                assert acc > 6

        plt.plot(node_counts, accuracy, '.:', label=f'x^{deg}')

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('node count')
    plt.suptitle(f'test quad gauss')
    plt.show()
