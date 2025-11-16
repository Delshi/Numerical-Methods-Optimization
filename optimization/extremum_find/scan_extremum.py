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
# epsilon = 0.000001


def R(x):
    return D * np.sin(A * (x**B) + C)


def ExtrFind(func, a, b, epsilon):
    x_ = []
    y_ = []
    step = epsilon / 2

    for x in np.arange(a, b + step, step):
        y = func(x)
        x_.append(x)
        y_.append(y)

    # * min(y_) for min(R) ; max(y_) for max(R)
    idx_max = y_.index(max(y_))

    x_max = x_[idx_max]
    y_max = y_[idx_max]

    return x_max, y_max


print("Экстремум на отрезке:", ExtrFind(R, a, b, epsilon), "\n")

x = np.arange(a, b, 0.01)
y = R(x)

x_max, y_max = ExtrFind(R, a, b, epsilon)

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
