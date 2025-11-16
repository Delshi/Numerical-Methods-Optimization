import numpy as np
import matplotlib.pyplot as plt

a = 5
b = 8.5
epsilon = 0.02
A = 1
B = 1
C = 5
D = 1


def R(x):
    return D * np.sin(A * (x**B) + C)


def extr_find(a, b, eps, delta, func):
    while (b - a) > eps:
        c = (a + b) / 2
        x1 = c - delta / 2
        x2 = c + delta / 2
        if func(x1) < func(x2):
            b = x2
        else:
            a = x1

    x_min = (a + b) / 2
    return x_min


delta = 1e-6

x_min = extr_find(a, b, epsilon, delta, R)

print(f"Найденный минимум: x = {x_min:.5f}, R(x) = {R(x_min):.5f}")

x_vals = np.linspace(a, b, 400)
y_vals = R(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="R(x)")
plt.scatter(
    [x_min],
    [R(x_min)],
    color="red",
    zorder=5,
    label=f"Минимум ({x_min:.5f}, {R(x_min):.5f})",
)
plt.xlabel("x")
plt.ylabel("R(x)")
plt.title("График функции и точка минимума")
plt.legend()
plt.grid(True)
plt.show()
