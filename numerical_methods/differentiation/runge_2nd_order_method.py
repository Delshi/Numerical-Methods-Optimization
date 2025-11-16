import numpy as np
import matplotlib.pyplot as plt


f_x = np.array(
    [2.10, 2.15, 2.20, 2.25, 2.30, 2.35, 2.40, 2.45, 2.50, 2.55], dtype=float
)
x = np.array(
    [
        0.86321,
        0.83690,
        0.80850,
        0.77807,
        0.74571,
        0.71147,
        0.67546,
        0.63776,
        0.59847,
        0.55768,
    ],
    dtype=float,
)


def InterpPoly(x):
    return (
        -4859.9405279
        + 62276.79703144 * x
        - 353510.71072187 * x**2
        + 1167419.18639214 * x**3
        - 2471792.47838595 * x**4
        + 3479849.37740984 * x**5
        - 3257470.54500256 * x**6
        + 1955176.40908357 * x**7
        - 682789.44057402 * x**8
        + 105703.69615607 * x**9
    )


def AnalytDeriv1(x):
    return (
        (12613727766 * x**8 / 13259)
        - (44867459719 * x**7 / 8214)
        + (254331302470 * x**6 / 18583)
        - (89104849288 * x**5 / 4559)
        + (76713279525 * x**4) / 4409
        - (130599627388 * x**3 / 13209)
        + (23643740782 * x**2 / 6751)
        - (226728336450 * x / 320681)
        + (5324417039 / 85496)
    )


def AnalytDeriv2(x):
    return (
        (100909822128 * x**7 / 13259)
        - (314072218033 * x**6 / 8214)
        + (1525987814820 * x**5 / 18583)
        - (445524246440 * x**4 / 4559)
        + (306853118100 * x**3 / 4409)
        - (130599627388 * x**2 / 4403)
        + (47287481564 * x / 6751)
        - (226728336450 / 320681)
    )


def NumericDeriv1(f, x_nodes, idx):
    if idx == 0:
        # Правая разность для первой точки
        h = x_nodes[idx + 1] - x_nodes[idx]
        return (
            -3 * f(x_nodes[idx]) + 4 * f(x_nodes[idx] + h) - f(x_nodes[idx] + 2 * h)
        ) / (2 * h)

    elif idx == len(x_nodes) - 1:
        # Левая разность для последней точки
        h = x_nodes[idx] - x_nodes[idx - 1]
        return (
            3 * f(x_nodes[idx]) - 4 * f(x_nodes[idx] - h) + f(x_nodes[idx] - 2 * h)
        ) / (2 * h)

    else:
        # Центральная разность для внутренних точек
        h_prev = x_nodes[idx] - x_nodes[idx - 1]
        h_next = x_nodes[idx + 1] - x_nodes[idx]

        # Взвешенная комбинация для неравномерной сетки
        return (
            h_prev**2 * f(x_nodes[idx + 1])
            - (h_prev**2 - h_next**2) * f(x_nodes[idx])
            - h_next**2 * f(x_nodes[idx - 1])
        ) / (h_prev * h_next * (h_prev + h_next))


def NumericDeriv2(f, x_nodes, idx):
    if idx == 0 or idx == len(x_nodes) - 1:
        # Для граничных точек односторонние разности
        if idx == 0:
            h = x_nodes[idx + 1] - x_nodes[idx]
            return (
                f(x_nodes[idx]) - 2 * f(x_nodes[idx] + h) + f(x_nodes[idx] + 2 * h)
            ) / h**2

        else:
            h = x_nodes[idx] - x_nodes[idx - 1]
            return (
                f(x_nodes[idx]) - 2 * f(x_nodes[idx] - h) + f(x_nodes[idx] - 2 * h)
            ) / h**2

    else:
        # Для внутренних точек
        h_prev = x_nodes[idx] - x_nodes[idx - 1]
        h_next = x_nodes[idx + 1] - x_nodes[idx]

        return (
            2
            * (
                h_prev * f(x_nodes[idx + 1])
                - (h_prev + h_next) * f(x_nodes[idx])
                + h_next * f(x_nodes[idx - 1])
            )
            / (h_prev * h_next * (h_prev + h_next))
        )


y_NumDeriv1 = np.array([NumericDeriv1(InterpPoly, x, i) for i in range(len(x))])
y_NumDeriv2 = np.array([NumericDeriv2(InterpPoly, x, i) for i in range(len(x))])

y_AnalytDeriv1 = AnalytDeriv1(x)
y_AnalytDeriv2 = AnalytDeriv2(x)


print("Таблица значений y'(x):")
print(" x\t\tЧисленно\tАналитически")
for xi, yp_num, yp_an in zip(x[::-1], y_NumDeriv1[::-1], y_AnalytDeriv1[::-1]):
    print(f"{xi:.5f}\t\t{yp_num:.4f}\t\t{yp_an:.4f}")


print("\nТаблица значений y''(x):")
print(" x\t\tЧисленно\tАналитически")
for xi, ydp_num, ydp_an in zip(x[::-1], y_NumDeriv2[::-1], y_AnalytDeriv2[::-1]):
    print(f"{xi:.5f}\t\t{ydp_num:.4f}\t\t{ydp_an:.4f}")


plt.style.use("seaborn-v0_8")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100)


ax1.plot(
    x,
    y_NumDeriv1,
    "o-",
    color="blue",
    markersize=6,
    linewidth=3,
    alpha=0.7,
    label="Численно",
    zorder=2,
)
ax1.plot(
    x,
    y_AnalytDeriv1,
    "--",
    color="orange",
    markersize=6,
    linewidth=2,
    label="Аналитически",
    zorder=2,
)

ax1.set_title("y'(x)", fontsize=14, pad=15)
ax1.set_xlabel("x", fontsize=8)
ax1.set_ylabel("y'", fontsize=8)
ax1.grid(True, linestyle=":", alpha=0.7)

legend1 = ax1.legend(fontsize=11, framealpha=0.95, shadow=True)
legend1.get_frame().set_facecolor("#f8f9fa")
legend1.get_frame().set_edgecolor("#dee2e6")


ax2.plot(
    x,
    y_NumDeriv2,
    "o-",
    color="red",
    markersize=6,
    linewidth=3,
    alpha=0.7,
    label="Численно",
    zorder=2,
)
ax2.plot(
    x,
    y_AnalytDeriv2,
    "--",
    color="green",
    markersize=6,
    linewidth=2,
    label="Аналитически",
    zorder=2,
)

ax2.set_title("y''(x)", fontsize=14, pad=15)
ax2.set_xlabel("x", fontsize=8)
ax2.set_ylabel("y''", fontsize=8)
ax2.grid(True, linestyle=":", alpha=0.7)

legend2 = ax2.legend(fontsize=11, framealpha=0.95, shadow=True)
legend2.get_frame().set_facecolor("#f8f9fa")
legend2.get_frame().set_edgecolor("#dee2e6")


plt.tight_layout()
plt.show()
