import matplotlib.pyplot as plt
import numpy as np

# x = np.array([0.23, 0.64, 1, 1.33, 1.89, 2, 3, 4.0123, 4.7843, 5, 6])
x = np.arange(0.1, 7, 0.15)
f_x = np.array(
    [
        (
            (x[i] ** 2 + np.sin(np.exp(x[i])) - np.sin(x[i]) / np.cos(x[i]))
            + 1 / np.sin(x[i])
        )
        for i in range(len(x))
    ]
)
c0 = x[0]
cn = x[-1]

z1, z2, z3, z4, z5, z6 = 0.1385, 0.4845, 2.334, 4.1, 5.015, 6.0094


def CubicSpline(x, f_x, c0, cn, target_point=None):
    n = x.size

    h = np.diff(x)
    h = np.insert(h, 0, 0)

    a = f_x[: n - 1].tolist()
    b, d = np.zeros(n), np.zeros(n)

    A = np.zeros([n - 2, n - 2])
    B = np.zeros([n - 2])

    for i in range(1, n - 1):
        # * Матрица коэффов системы
        if i == 1:
            A[i - 1, i - 1] = 2 * (h[i] + h[i + 1])
            A[i - 1, i] = h[i + 1]

        elif i + 1 == n - 1:
            A[i - 1, i - 2] = h[i]
            A[i - 1, i - 1] = 2 * (h[i] + h[i + 1])

        else:
            A[i - 1, i - 2] = h[i]
            A[i - 1, i - 1] = 2 * (h[i] + h[i + 1])
            A[i - 1, i] = h[i + 1]

        # * Матрица свободных членов
        B[i - 1] = 3 * (
            ((f_x[i + 1] - f_x[i]) / h[i + 1]) - (([f_x[i] - f_x[i - 1]]) / h[i])
        )

    C = np.linalg.solve(A, B)
    C = np.insert(C, 0, c0)
    C = np.insert(C, C.size, cn)

    # * Коэффы d и b
    for i in range(0, n - 1):
        d[i] = (C[i + 1] - C[i]) / (3 * h[i + 1])
        b[i] = ((f_x[i + 1] - f_x[i]) / h[i + 1]) - (
            ((C[i + 1] + 2 * C[i]) * h[i + 1]) / 3
        )

    # * Массивы значений, полученные аппроксимацией (для построения сплайнов)
    x_appr = np.array([])
    for i in range(x.size - 1):
        x_interval = np.linspace(x[i], x[i + 1], 20)
        x_appr = np.concatenate((x_appr, x_interval))

    f_x_appr = []
    for x_point in x_appr:
        k = max(0, min(np.searchsorted(x, x_point) - 1, len(x) - 2))
        dx = x_point - x[k]
        S_k = a[k] + b[k] * dx + C[k] * (dx**2) + d[k] * (dx**3)
        f_x_appr.append(S_k)

    # * Аппроксимация значения функции в точке
    if target_point is not None:
        if target_point < x[0] or target_point > x[-1]:
            raise ValueError(f"Точка {target_point} вне диапазона аппроксимации")

        k = np.searchsorted(x, target_point) - 1

        if k == x.size - 1:
            k -= 1

        dx = target_point - x[k]

        f_target_point = a[k] + b[k] * dx + C[k] * (dx**2) + d[k] * (dx**3)

        return x_appr, f_x_appr, f_target_point

    else:
        return np.array(x_appr), np.array(f_x_appr)


x_appr, f_x_appr = CubicSpline(x, f_x, c0, cn)
_, __, f_z1 = CubicSpline(x, f_x, c0, cn, z1)
_, __, f_z2 = CubicSpline(x, f_x, c0, cn, z2)
_, __, f_z3 = CubicSpline(x, f_x, c0, cn, z3)
_, __, f_z4 = CubicSpline(x, f_x, c0, cn, z4)
_, __, f_z5 = CubicSpline(x, f_x, c0, cn, z5)
_, __, f_z6 = CubicSpline(x, f_x, c0, cn, z6)


print(
    "Сплайн\n", f_z1, "\n", f_z2, "\n", f_z3, "\n", f_z4, "\n", f_z5, "\n", f_z6, "\n"
)
print(
    "Numpy\n",
    ((z1**2 + np.sin(np.exp(z1)) - np.sin(z1) / np.cos(z1)) + 1 / np.sin(z1)),
    "\n",
    ((z2**2 + np.sin(np.exp(z2)) - np.sin(z2) / np.cos(z2)) + 1 / np.sin(z2)),
    "\n",
    ((z3**2 + np.sin(np.exp(z3)) - np.sin(z3) / np.cos(z3)) + 1 / np.sin(z3)),
    "\n",
    ((z4**2 + np.sin(np.exp(z4)) - np.sin(z4) / np.cos(z4)) + 1 / np.sin(z4)),
    "\n",
    ((z5**2 + np.sin(np.exp(z5)) - np.sin(z5) / np.cos(z5)) + 1 / np.sin(z5)),
    "\n",
    ((z6**2 + np.sin(np.exp(z6)) - np.sin(z6) / np.cos(z6)) + 1 / np.sin(z6)),
    "\n",
)


plt.figure(figsize=(10, 6))
plt.plot(x, f_x, "o", markersize=4, label="Исходные узлы", color="black")
plt.plot(x_appr, f_x_appr, "-", linewidth=3, label="Кубический сплайн", color="purple")
plt.title("Интерполяция кубическим сплайном")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.scatter(
    z1, f_z1, color="darkorange", marker="o", s=50, label=f"z1 = {z1}, f(z1) = {f_z1}"
)
plt.scatter(
    z2, f_z2, color="darkorange", marker="o", s=50, label=f"z2 = {z2}, f(z2) = {f_z2}"
)
plt.scatter(
    z3, f_z3, color="darkorange", marker="o", s=50, label=f"z3 = {z3}, f(z3) = {f_z3}"
)
plt.scatter(
    z4, f_z4, color="darkorange", marker="o", s=50, label=f"z4 = {z4}, f(z4) = {f_z4}"
)
plt.scatter(
    z5, f_z5, color="darkorange", marker="o", s=50, label=f"z5 = {z5}, f(z5) = {f_z5}"
)
plt.scatter(
    z6, f_z6, color="darkorange", marker="o", s=50, label=f"z6 = {z6}, f(z6) = {f_z6}"
)
plt.legend()
plt.grid(True)
plt.show()
