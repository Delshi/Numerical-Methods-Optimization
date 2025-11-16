import math
import numpy as np
import matplotlib.pyplot as plt

a, b = 5, 8.5
A, B, C, D = 1, 1, 5, 1
epsilon = 0.02
# epsilon = 0.0000002


def R(x):
    return D * np.sin(A * (x**B) + C)


def ExtrFind(f, a, b, epsilon):
    iteration = 0
    while (b - a) > epsilon:
        x1 = a + (b - a) / 4
        x2 = a + (b - a) / 2
        x3 = a + 3 * (b - a) / 4

        f1, f2, f3 = f(x1), f(x2), f(x3)

        # Коэффициенты кубической интерполяции
        denom = (x2 - x1) * (x3 - x1) * (x3 - x2)
        A_interp = (x3 * (f2 - f1) + x2 * (f1 - f3) + x1 * (f3 - f2)) / denom
        B_interp = (
            (x3**2) * (f1 - f2) + (x2**2) * (f3 - f1) + (x1**2) * (f2 - f3)
        ) / denom
        C_interp = (
            x2 * x3 * (x2 - x3) * f1
            + x3 * x1 * (x3 - x1) * f2
            + x1 * x2 * (x1 - x2) * f3
        ) / denom

        vertex = -B_interp / (2 * A_interp)
        f_vertex = f(vertex)

        if vertex < x2:
            if f_vertex < f2:
                b = x2
            else:
                a = vertex
        else:
            if f_vertex < f2:
                a = x2
            else:
                b = vertex

    return vertex


result = ExtrFind(R, a, b, epsilon)
print(f"Найденный экстремум: x = {result}, R(x) = {R(result)}")
x_max = result
y_max = R(x_max)

x = np.arange(a, b, 0.01)
y = R(x)

plt.plot(x, y, label="R(x)", linewidth=2, color="purple")
plt.xlabel("x")
plt.ylabel("R(x)")
plt.scatter(x_max, y_max, marker="X", c="darkorange", label="Экстремум", linewidth=2)
plt.annotate(
    f"({x_max:.4f}, {y_max:.4f})", (x_max, y_max), xytext=(x_max - 0.5, y_max + 0.1)
)
plt.axvline(a, color="black", linestyle="--", label="Границы [a, b]", linewidth=2)
plt.axvline(b, color="black", linestyle="--", linewidth=2)
plt.legend()
plt.title(f"R(x) на отрезке [{a}, {b}]")
plt.grid(True)
plt.show()
