import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


print("Функция А\n")


x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
f_x = np.array(
    [
        0.946083,
        1.028685,
        1.108047,
        1.183958,
        1.256227,
        1.324684,
        1.389181,
        1.449592,
        1.505817,
        1.557775,
    ]
)

points = [
    [0.175418, 0.715878, 0.464331, 1.344273],
    [0.090566, 0.826611, 0.395142, 1.534355],
    [0.157369, 0.826216, 0.445135, 1.273228],
    [0.053526, 0.866027, 0.610308, 1.435993],
    [0.109658, 0.710092, 0.468515, 1.026087],
    [0.094189, 0.755627, 0.343888, 1.058530],
    [0.132747, 0.755108, 0.447232, 1.390474],
    [0.058421, 0.731564, 0.429815, 1.417970],
    [0.063946, 0.907395, 0.455020, 1.373809],
    [0.182461, 0.717093, 0.525286, 1.197522],
    [0.173762, 0.759231, 0.482714, 1.470054],
    [0.129140, 0.766507, 0.238812, 1.183944],
]


def lagrange_interpolation(x_values, y_values, x):
    n = len(x_values)
    result = 0.0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result


lagrange_results = []
for i, (t1, t2, t3, a) in enumerate(points):
    f_t1_lagrange = lagrange_interpolation(x, f_x, t1)
    f_t2_lagrange = lagrange_interpolation(x, f_x, t2)
    f_t3_lagrange = lagrange_interpolation(x, f_x, t3)
    lagrange_results.append(
        (i + 1, t1, t2, t3, a, f_t1_lagrange, f_t2_lagrange, f_t3_lagrange)
    )


print("Лагранж:")
for result in lagrange_results:
    print(
        f"N={result[0]}: t1={result[1]:.6f}, f(t1)={result[5]:.6f}; t2={result[2]:.6f}, f(t2)={result[6]:.6f}; t3={result[3]:.6f}, f(t3)={result[7]:.6f}; a={result[4]:.6f}"
    )


interp_func = interp1d(x, f_x, kind="linear", fill_value="extrapolate")

scipy_results = []

for i, (t1, t2, t3, a) in enumerate(points):
    f_t1_scipy = interp_func(t1)
    f_t2_scipy = interp_func(t2)
    f_t3_scipy = interp_func(t3)
    scipy_results.append((i + 1, t1, t2, t3, a, f_t1_scipy, f_t2_scipy, f_t3_scipy))


print("\nSciPy:")
for result in scipy_results:
    print(
        f"N={result[0]}: t1={result[1]:.6f}, f(t1)={result[5]:.6f}; t2={result[2]:.6f}, f(t2)={result[6]:.6f}; t3={result[3]:.6f}, f(t3)={result[7]:.6f}; a={result[4]:.6f}"
    )


print("\nЛагранж - эталонное значение:")
for i, (t1, t2, t3, a) in enumerate(points):
    f_t1_lagrange = lagrange_interpolation(x, f_x, t1)
    f_t2_lagrange = lagrange_interpolation(x, f_x, t2)
    f_t3_lagrange = lagrange_interpolation(x, f_x, t3)

    diff_t1 = abs(a - f_t1_lagrange)
    diff_t2 = abs(a - f_t2_lagrange)
    diff_t3 = abs(a - f_t3_lagrange)

    print(
        f"N={i+1}: a={a:.6f}, diff_t1={diff_t1:.6f}, diff_t2={diff_t2:.6f}, diff_t3={diff_t3:.6f}"
    )


plt.figure(figsize=(12, 8))
plt.plot(x, f_x, "bo-", label="Табличная функция", markersize=8)

point_size = 2
for i, (t1, t2, t3, a) in enumerate(points):
    plt.plot(
        t1,
        lagrange_interpolation(x, f_x, t1),
        "ro",
        label=f"t1, Лагранж" if i == 0 else "",
        markersize=point_size + 2,
    )
    plt.plot(
        t2,
        lagrange_interpolation(x, f_x, t2),
        "go",
        label=f"t2, Лагранж" if i == 0 else "",
        markersize=point_size + 2,
    )
    plt.plot(
        t3,
        lagrange_interpolation(x, f_x, t3),
        "mo",
        label=f"t3, Лагранж" if i == 0 else "",
        markersize=point_size + 2,
    )

for i, (t1, t2, t3, a) in enumerate(points):
    plt.plot(
        t1,
        interp_func(t1),
        "gx",
        label=f"t1, SciPy" if i == 0 else "",
        markersize=point_size + 1,
    )
    plt.plot(
        t2,
        interp_func(t2),
        "rx",
        label=f"t2, SciPy" if i == 0 else "",
        markersize=point_size + 1,
    )
    plt.plot(
        t3,
        interp_func(t3),
        "bx",
        label=f"t3, SciPy" if i == 0 else "",
        markersize=point_size + 1,
    )


plt.title("Лагранж и SciPy")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 1), loc="upper left"
)

plt.tight_layout()
plt.show()
