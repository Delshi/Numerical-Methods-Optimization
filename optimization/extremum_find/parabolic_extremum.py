import math
import numpy as np
import matplotlib.pyplot as plt

a, b = 5, 8.5
A, B, C, D = 1, 1, 5, 1
epsilon = 0.02


def R(x):
    return D * np.sin(A * (x**B) + C)


def ExtrFind(func, a, b, c, epsilon, max_iter=10000):
    iteration = 0
    while abs(c - a) > epsilon and iteration < max_iter:
        a_, b_, c_ = func(a), func(b), func(c)
        denominator = (b - c) * a_ + (c - a) * b_ + (a - b) * c_

        if abs(denominator) < 1e-12:
            # БМВ, поэтому чутка добавляем смещение
            a += 1e-12
            c -= 1e-12
            continue

        x = (
            0.5
            * ((b**2 - c**2) * a_ + (c**2 - a**2) * b_ + (a**2 - b**2) * c_)
            / denominator
        )

        if not (a < x < c):
            x = (a + c) / 2

        if func(x) < func(b):
            if x > b:
                a, b = b, x
            else:
                c, b = b, x
        else:
            if x > b:
                c = x
            else:
                a = x

        iteration += 1

    if iteration == max_iter:
        print("Конец итераций")

    return b, func(b)


x_max, y_max = ExtrFind(R, a, (a + b) / 2, b, epsilon)
print("Экстремум на отрезке:", (x_max, y_max), "\n")

x = np.arange(a, b, 0.01)
y = R(x)

plt.plot(x, y, label="R(x)", linewidth=2, color="purple")
plt.xlabel("x")
plt.ylabel("R(x)")
plt.scatter(x_max, y_max, marker="X", c="darkorange", label="Экстремум", linewidth=2)
plt.annotate(f"({x_max:.4f},{y_max:.4f})", (x_max, y_max), xytext=(x_max - 0.9, y_max))
plt.axvline(a, color="black", linestyle="--", label="[a, b]", linewidth=2)
plt.axvline(b, color="black", linestyle="--", linewidth=2)
plt.legend()
plt.title(f"R(x), [{a}, {b}]")
plt.grid(True)
plt.show()
