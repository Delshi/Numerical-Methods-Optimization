import math
import numpy as np
import matplotlib.pyplot as plt

a = 5
b = 8.5

A = 1
B = 1
C = 5
D = 1

epsilon = 0.02
# epsilon = 0.0000000001

phi = 0.5 * (1 + np.sqrt(5))


def R(x):
    return D * np.sin(A * (x**B) + C)


def rootFind(func, a, b, epsilon):
    x1 = b - ((b - a) / phi)
    x2 = a + ((b - a) / phi)

    while np.abs(b - a) > epsilon:

        if R(x1) < R(x2):
            b = x2
            x2 = x1
            x1 = b - ((b - a) / phi)

        else:
            a = x1
            x1 = x2
            x2 = a + ((b - a) / phi)

    x_max = (a + b) / 2
    y_max = R(x_max)

    return x_max, y_max


print("Экстремум на отрезке:", rootFind(R, a, b, epsilon), "\n")

x = np.arange(a, b, 0.01)
y = R(x)

x_max, y_max = rootFind(R, a, b, epsilon)

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
