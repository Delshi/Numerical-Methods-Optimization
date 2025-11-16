import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt


def lagrange_interpolation(x, y, x_point):
    """Интерполяция методом Лагранжа"""
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_point - x[j]) / (x[i] - x[j])
        result += term
    return result


def divided_differences(x, y):
    """Вычисление разделённых разностей для метода Ньютона"""
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    return coef[0]


def newton_interpolation(x, y, x_point):
    """Интерполяция методом Ньютона"""
    coef = divided_differences(x, y)
    n = len(coef)
    result = coef[0]
    product = 1.0

    for i in range(1, n):
        product *= x_point - x[i - 1]
        result += coef[i] * product

    return result


x_data = np.array([0, 1, 2, 6], dtype=float)
y_data = np.array([-1, -3, 3, 1187], dtype=float)
target_x = 4.0  # Точка, в которой ищем значение

lagrange_result = lagrange_interpolation(x_data, y_data, target_x)
newton_result = newton_interpolation(x_data, y_data, target_x)

poly = lagrange(x_data, y_data)
scipy_result = poly(target_x)

print(f"Метод Лагранжа: f({target_x}) = {lagrange_result:.4f}")
print(f"Метод Ньютона:  f({target_x}) = {newton_result:.4f}")
print(f"SciPy проверка: f({target_x}) = {scipy_result:.4f}")
print(f"Разница между методами: {abs(lagrange_result - newton_result):.2e}")

x_plot = np.linspace(min(x_data), max(x_data), 100)
y_lagrange = [lagrange_interpolation(x_data, y_data, x) for x in x_plot]
y_newton = [newton_interpolation(x_data, y_data, x) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, "ro", label="Исходные точки")
plt.plot(x_plot, y_lagrange, "-", label="Лагранж", color="purple", linewidth=3)
plt.plot(x_plot, y_newton, ":", label="Ньютон", color="orange", linewidth=3)
plt.plot(target_x, lagrange_result, "ks", markersize=8, label=f"f({target_x})")
plt.title("Сравнение методов интерполяции")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()
