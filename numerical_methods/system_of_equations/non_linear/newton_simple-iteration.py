import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def Sys1(x, y):
    x_new = 0.8 - np.cos(y + 0.5)
    y_new = (np.sin(x) - 1.6) / 2

    return x_new, y_new


def Sys2Minus(x, y):
    x_new = np.sin(x + y) / 1.4
    y_new = np.sqrt(1 - x**2)

    return x_new, y_new


def Sys2Plus(x, y):
    x_new = np.sin(x + y) / 1.4
    y_new = -np.sqrt(1 - x**2)

    return x_new, y_new


def SimpleIter(x0, y0, system, eps=0.001, max_iter=1000):
    x, y = x0, y0

    for i in range(max_iter):
        x_new, y_new = system(x, y)

        if np.abs(x_new - x) < eps and np.abs(y_new - y) < eps:
            break

        x, y = x_new, y_new

    return x_new, y_new


def NewtonSolve(system, x0, y0, eps=0.001, max_iter=100):

    def equations(vars):
        x, y = vars
        f1, f2 = system(x, y)
        return [f1 - x, f2 - y]

    def jacobian(vars):
        x, y = vars
        h = 1e-6

        J = np.zeros((2, 2))

        # df1/dx
        f1, _ = system(x + h, y)
        f1_0, _ = system(x, y)
        J[0, 0] = (f1 - f1_0) / h - 1

        # df1/dy
        _, f1 = system(x, y + h)
        _, f1_0 = system(x, y)
        J[0, 1] = (f1 - f1_0) / h

        # df2/dx
        f2, _ = system(x + h, y)
        f2_0, _ = system(x, y)
        J[1, 0] = (f2 - f2_0) / h

        # df2/dy
        _, f2 = system(x, y + h)
        _, f2_0 = system(x, y)
        J[1, 1] = (f2 - f2_0) / h - 1

        return J

    x, y = x0, y0

    for i in range(max_iter):
        F = np.array(equations([x, y]))
        J = jacobian([x, y])

        delta = np.linalg.solve(J, -F)
        x += delta[0]
        y += delta[1]

        if np.linalg.norm(delta) < eps:
            break

    return x, y


x0, y0 = 0, 0


Sys1_X_Sol, Sys1_Y_Sol = SimpleIter(x0, y0, Sys1)
Sys2_X_Sol_Plus, Sys2_Y_Sol_Plus = SimpleIter(x0, y0, Sys2Plus)
Sys2_X_Sol_Minus, Sys2_Y_Sol_Minus = SimpleIter(x0, y0, Sys2Minus)


Newton_Sys1_X, Newton_Sys1_Y = NewtonSolve(Sys1, x0, y0)
Newton_Sys2_X_Plus, Newton_Sys2_Y_Plus = NewtonSolve(Sys2Plus, x0, y0)
Newton_Sys2_X_Minus, Newton_Sys2_Y_Minus = NewtonSolve(Sys2Minus, x0, y0)


print("\nМетод итераций:")
print("а)", (float("{:.8f}".format(Sys1_X_Sol)), float("{:.8f}".format(Sys1_Y_Sol))))
print(
    "б) (+):",
    (float("{:.8f}".format(Sys2_X_Sol_Plus)), float("{:.8f}".format(Sys2_Y_Sol_Plus))),
)
print(
    "б) (-):",
    (
        float("{:.8f}".format(Sys2_X_Sol_Minus)),
        float("{:.8f}".format(Sys2_Y_Sol_Minus)),
    ),
)

print("\nМетод Ньютона:")
print(
    "а)", (float("{:.8f}".format(Newton_Sys1_X)), float("{:.8f}".format(Newton_Sys1_Y)))
)
print(
    "б) (+):",
    (
        float("{:.8f}".format(Newton_Sys2_X_Plus)),
        float("{:.8f}".format(Newton_Sys2_Y_Plus)),
    ),
)
print(
    "б) (-):",
    (
        float("{:.8f}".format(Newton_Sys2_X_Minus)),
        float("{:.8f}".format(Newton_Sys2_Y_Minus)),
    ),
)
print("\n")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)
F1 = np.cos(Y + 0.5) + X - 0.8
F2 = np.sin(X) - 2 * Y - 1.6


ax1.contour(X, Y, F1, levels=[0], colors="darkblue", linewidths=2)
ax1.contour(X, Y, F2, levels=[0], colors="red", linewidths=2)


ax1.plot(
    Sys1_X_Sol,
    Sys1_Y_Sol,
    "o",
    markersize=8,
    markeredgecolor="black",
    markerfacecolor="orange",
    label=f"Решение: ({Sys1_X_Sol:.4f}, {Sys1_Y_Sol:.4f})",
)


ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("y", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.7)
ax1.set_title(f"a): {{cos(y+0.5)+x=0.8 ; sin(x)-2y=1.6}}", pad=15)
ax1.legend()


x = np.linspace(-1.5, 1.5, 500)
y = np.linspace(-1.5, 1.5, 500)
X, Y = np.meshgrid(x, y)
F1 = np.sin(X + Y) - 1.4 * X
F2 = X**2 + Y**2 - 1


ax2.contour(X, Y, F1, levels=[0], colors="darkblue", linewidths=2)
ax2.contour(X, Y, F2, levels=[0], colors="red", linewidths=2)


ax2.plot(
    Sys2_X_Sol_Plus,
    Sys2_Y_Sol_Plus,
    "o",
    markersize=8,
    markeredgecolor="black",
    markerfacecolor="orange",
    label=f"Решение (+): ({Sys2_X_Sol_Plus:.4f}, {Sys2_Y_Sol_Plus:.4f})",
)
ax2.plot(
    Sys2_X_Sol_Minus,
    Sys2_Y_Sol_Minus,
    "o",
    markersize=8,
    markeredgecolor="black",
    markerfacecolor="lime",
    label=f"Решение (-): ({Sys2_X_Sol_Minus:.4f}, {Sys2_Y_Sol_Minus:.4f})",
)


ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("y", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.7)
ax2.set_title(f"б): {{sin(x+y)-1.4x=0 и x²+y²=1}}", pad=15)
ax2.axis("equal")
ax2.legend()


plt.tight_layout()
plt.show()

input()
