import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def Func(x, y):
    return x + np.sin(y / np.sqrt(10))


def SolveDiff(func, x0, y0, interval, step, method):
    a, b = interval
    h = step
    a, b = interval
    x = np.arange(a, b + step, step)
    y = np.zeros_like(x)
    y[0] = y0

    def Euler(func, x, y, idx):
        return y[idx - 1] + (x[idx] - x[idx - 1]) * func(x[idx - 1], y[idx - 1])

    def Runge(func, x, y, idx):
        xn = x[idx - 1]
        yn = y[idx - 1]

        k1 = func(xn, yn)
        k2 = func(xn + h / 2, yn + h / 2 * k1)
        k3 = func(xn + h / 2, yn + h / 2 * k2)
        k4 = func(xn + h, yn + h * k3)

        return yn + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def AdamsRunge(func, x, y, idx):
        if idx >= 4:
            f0 = func(x[idx - 1], y[idx - 1])
            f1 = func(x[idx - 2], y[idx - 2])
            f2 = func(x[idx - 3], y[idx - 3])
            f3 = func(x[idx - 4], y[idx - 4])
            return y[idx - 1] + h * (55 * f0 - 59 * f1 + 37 * f2 - 9 * f3) / 24
        else:
            return Runge(func, x, y, idx)

    def AdamsEuler(func, x, y, idx):
        if idx >= 4:
            f0 = func(x[idx - 1], y[idx - 1])
            f1 = func(x[idx - 2], y[idx - 2])
            f2 = func(x[idx - 3], y[idx - 3])
            f3 = func(x[idx - 4], y[idx - 4])
            return y[idx - 1] + h * (55 * f0 - 59 * f1 + 37 * f2 - 9 * f3) / 24
        else:
            return Euler(func, x, y, idx)

    solver = {
        "Euler": Euler,
        "Runge": Runge,
        "Adams-Euler": AdamsEuler,
        "Adams-Runge": AdamsRunge,
    }.get(method)

    for i in range(1, len(x)):
        y[i] = solver(func, x, y, i)

    return x, y


step = 0.1
interval = (0.6, 1.6)
x0, y0 = 0.6, 0.8


xR, yR = SolveDiff(Func, x0=x0, y0=y0, interval=interval, step=step, method="Runge")
xAR, yAR = SolveDiff(
    Func, x0=x0, y0=y0, interval=interval, step=step, method="Adams-Runge"
)
xAE, yAE = SolveDiff(
    Func, x0=x0, y0=y0, interval=interval, step=step, method="Adams-Euler"
)
xE, yE = SolveDiff(Func, x0=x0, y0=y0, interval=interval, step=step, method="Euler")
DiffSolutionSciPy = solve_ivp(
    Func, interval, [y0], t_eval=np.arange(interval[0], interval[1] + step, step)
)


print("\nЭйлер:")
print([f"({x:.6f}, {y:.6f})" for x, y in zip(xE, yE)])
print("\nРунге-Кутта:")
print([f"({x:.6f}, {y:.6f})" for x, y in zip(xR, yR)])
print("\nАдамс-Рунге:")
print([f"({x:.6f}, {y:.6f})" for x, y in zip(xAR, yAR)])
print("\nАдамс-Эйлер:")
print([f"({x:.6f}, {y:.6f})" for x, y in zip(xAE, yAE)])


plt.figure(figsize=(10, 6))

plt.plot(xE, yE, "o-", label="Эйлер", markersize=5, color="crimson")
plt.plot(xR, yR, "o-", label="Рунге-Кутта", markersize=6, color="orange", linewidth=4)
plt.plot(xAR, yAR, "o-", label="Адамс-Рунге", markersize=3, color="lime")
plt.plot(xAE, yAE, "o-", label="Адамс-Эйлер", markersize=5, color="darkblue")
plt.plot(
    DiffSolutionSciPy.t,
    DiffSolutionSciPy.y[0],
    "--.",
    label="SciPy",
    markersize=3,
    color="black",
)

plt.title("y' = x + sin(y/√10)")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True)

plt.show()
