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


def FibonacciNums(n):
    if n <= 1:
        return n
    else:
        return FibonacciNums(n - 1) + FibonacciNums(n - 2)


def ExtrFind(func, a, b, epsilon):
    n = 0
    while FibonacciNums(n) < (b - a) / epsilon:
        n += 1

    x1 = a + (FibonacciNums(n - 2) / FibonacciNums(n)) * (b - a)
    x2 = a + (FibonacciNums(n - 1) / FibonacciNums(n)) * (b - a)

    f1 = func(x1)
    f2 = func(x2)

    for i in range(2, n):
        # * "<" for min; ">" for max
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (FibonacciNums(n - i) / FibonacciNums(n - i + 2)) * (b - a)
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (FibonacciNums(n - i + 1) / FibonacciNums(n - i + 2)) * (b - a)
            f2 = func(x2)

    x_max = (a + b) / 2
    y_max = R(x_max)

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
