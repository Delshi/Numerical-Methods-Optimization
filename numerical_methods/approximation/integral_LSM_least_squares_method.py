import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 20, 1) * 0.1
y = np.array(
    [
        0.08,
        0.14,
        0.37,
        0.36,
        0.44,
        0.48,
        0.27,
        0.39,
        0.50,
        0.48,
        0.69,
        0.50,
        0.31,
        0.37,
        0.43,
        0.33,
        0.31,
        0.09,
        0.08,
        0.03,
    ]
)


def LSMPoly(a, b, c, x):
    y_approx = []
    for i in range(x.size):
        y_approx.append(a * (x[i]) ** 2 + b * x[i] + c)

    return y_approx


def integrate(f, x):
    return np.trapz(f, x)


A = np.array(
    [
        [integrate(x**4, x), integrate(x**3, x), integrate(x**2, x)],
        [integrate(x**3, x), integrate(x**2, x), integrate(x, x)],
        [integrate(x**2, x), integrate(x, x), integrate(np.ones_like(x), x)],
    ]
)


B = np.array([integrate(y * x**2, x), integrate(y * x, x), integrate(y, x)])


X = np.linalg.solve(A, B)
print(f"\na = {X[0]:.8f} \nb = {X[1]:.8f} \nc = {X[2]:.8f}")


y_approx = LSMPoly(X[0], X[1], X[2], x)

S = np.sqrt(np.sum([(y_approx[i] - y[i]) ** 2 for i in range(y.size)]))

print(f"\nСКО: {S:.8f}")


coeffs_np = np.polyfit(x, y, 2)
poly_np = np.poly1d(coeffs_np)


plt.figure(figsize=(12, 7), dpi=100)
plt.style.use("seaborn-v0_8")

plt.plot(
    x,
    y,
    "o-",
    color="red",
    markersize=5,
    linewidth=1,
    alpha=0.7,
    label="Исходные точки",
    zorder=3,
)

plt.plot(
    x,
    y_approx,
    "--",
    color="orange",
    markersize=6,
    linewidth=2,
    label=f"LSMPoly: {X[0]:.4f}x² + {X[1]:.4f}x + {X[2]:.4f}",
    zorder=2,
)

plt.plot(
    x,
    poly_np(x),
    ".-",
    color="darkblue",
    linewidth=3,
    label=f"Numpy polyfit: {coeffs_np[0]:.4f}x² + {coeffs_np[1]:.4f}x + {coeffs_np[2]:.4f}",
    zorder=1,
)

plt.title("Least Squares Method", fontsize=14, pad=15)
plt.xlabel("X", fontsize=8)
plt.ylabel("Y", fontsize=8)
plt.grid(True, linestyle=":", alpha=0.7)

legend = plt.legend(fontsize=11, framealpha=0.95, shadow=True)
legend.get_frame().set_facecolor("#f8f9fa")
legend.get_frame().set_edgecolor("#dee2e6")

plt.tight_layout()
plt.show()
