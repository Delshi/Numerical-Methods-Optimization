import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def TMA(x, h):
    n = len(x) - 1

    A = np.zeros(n + 1)
    B = np.zeros(n + 1)
    C = np.zeros(n + 1)
    F = np.zeros(n + 1)

    # Внутренние коэффы
    for i in range(1, n):
        A[i] = 1 - x[i] * h
        B[i] = -2 - 2 * h**2
        C[i] = 1 + x[i] * h
        F[i] = 0.6 * h**2

    # Краевые условия (i=0 и i=n)
    B[0] = -1
    C[0] = 1
    F[0] = h

    A[n] = 1
    B[n] = 0.4 * h - 1
    F[n] = h

    # Прямой ход прогонки (вычисление alpha и beta)
    alpha = np.zeros(n + 1)
    beta = np.zeros(n + 1)

    alpha[0] = -C[0] / B[0]
    beta[0] = F[0] / B[0]

    for i in range(1, n + 1):
        denominator = B[i] + A[i] * alpha[i - 1]
        if i < n:
            alpha[i] = -C[i] / denominator
        beta[i] = (F[i] - A[i] * beta[i - 1]) / denominator

    # Обратный ход прогонки (нахождение у)
    y = np.zeros(n + 1)
    y[n] = beta[n]

    for i in range(n - 1, -1, -1):
        y[i] = alpha[i] * y[i + 1] + beta[i]

    return y


h = 0.005
a, b = 2, 2.3
x = np.arange(a, b + h, h)
y = TMA(x, h)


print(f"\ny({x[0]:.4}) = {y[0]:.8} ; y({x[-1]:.4}) = {y[-1]:.8}\n")


plt.style.use("seaborn-v0_8")
plt.figure(figsize=(9, 12))
title = (
    r"$y''+2xy'-2y = 0.6$" + "\n" + r"$y'(2) = 1$" + "\n" + r"$0.4y(2.3) - y'(2.3) = 1$"
)
plt.plot(x, y, "o-", markersize=5, label=f"y(x)", color="crimson")
plt.xlabel("x", fontsize=12)
plt.ylabel("y(x)", fontsize=12)
plt.title(title, fontsize=14, pad=20)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.legend(fontsize=12, framealpha=1, shadow=True)
plt.tight_layout()
plt.show()
