import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange


def interp_lagrange(x, y, t):
    z = 0

    for j in range(len(y)):
        p1 = 1
        p2 = 1

        for i in range(len(x)):
            if i != j:
                p1 = p1 * (t - x[i])
                p2 = p2 * (x[j] - x[i])

        z = z + y[j] * p1 / p2

    return z


y = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)
x = np.array(
    [1, 0.995, 0.9801, 0.9553, 0.9211, 0.8756, 0.8553, 0.7648, 0.6967], dtype=float
)

x_new = np.linspace(np.min(x), np.max(x), 100)
y_new = [interp_lagrange(x, y, i) for i in x_new]

poly = lagrange(x, y)

y_scipy = poly(x_new)

print("Интерполяционный многочлен SciPy:\n", poly)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x, y, "ro", label="Исходные точки")
plt.plot(x_new, y_new, "b-", label="Самодельный метод")
plt.title("Самодельная интерполяция Лагранжа")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y, "ro", label="Исходные точки")
plt.plot(x_new, y_scipy, "g--", label="SciPy метод")
plt.title(f"Интерполяция через SciPy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
