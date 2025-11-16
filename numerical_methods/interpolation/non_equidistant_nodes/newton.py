import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import KroghInterpolator
from numpy.polynomial.polynomial import Polynomial


def divided_differences(x, y):
    """Вычисляет таблицу разделённых разностей"""
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    return coef[0, :]


def interp_newton(x, y, t):
    """Интерполяционный многочлен Ньютона"""
    coefficients = divided_differences(x, y)
    result = coefficients[0]
    product_term = 1.0

    for i in range(1, len(x)):
        product_term *= t - x[i - 1]
        result += coefficients[i] * product_term

    return result


y = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)
x = np.array(
    [1, 0.995, 0.9801, 0.9553, 0.9211, 0.8756, 0.8553, 0.7648, 0.6967], dtype=float
)
x_new = np.linspace(np.min(x), np.max(x), 100)

y_newton = [interp_newton(x, y, t) for t in x_new]
krogh_interp = KroghInterpolator(x, y)
y_scipy = krogh_interp(x_new)

# Получение коэффициентов через значения в точках
x_vals = np.linspace(
    0, 1, len(x)
)  # Произвольные точки для построения системы уравнений
A = np.vander(x_vals, increasing=True)  # Матрица Вандермонда
b = krogh_interp(x_vals)
coefficients = np.linalg.solve(A, b)  # Решение системы уравнений
poly = Polynomial(coefficients)

# Вывод результатов
print("Полином в степенной форме:")
print(poly)

plt.figure(figsize=(14, 6))

# График метода Ньютона
plt.subplot(1, 2, 1)
plt.plot(x, y, "ro", label="Исходные точки")
plt.plot(x_new, y_newton, "b-", label="Самодельный Ньютон")
plt.title("Самодельная реализация Ньютона")
plt.grid(True)
plt.legend()

# График SciPy (Krogh)
plt.subplot(1, 2, 2)
plt.plot(x, y, "ro", label="Исходные точки")
plt.plot(x_new, y_scipy, "g--", label="SciPy (Krogh)")
plt.title("Интерполяция через SciPy\nKroghInterpolator")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
