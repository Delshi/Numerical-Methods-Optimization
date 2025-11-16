import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A, B, C, D = 2, 3, 4, 4
x1_start, x1_stop = -2, 2
x2_start, x2_stop = -2, 2
initial_point = np.array([-1, 1])
eps = 0.01
g = 0.01
step = 0.1


def TargetFunc(x1, x2):
    return A * x1**3 + B * x2**3 - C * x1 - D * x2


def gradient_descent_pairwise(f, initial_point, eps, g, step, max_iter=100):
    x_k = np.array(initial_point, dtype=np.float64)
    trajectory = [x_k.copy()]
    for _ in range(max_iter):
        u = np.random.uniform(-1, 1, size=2)
        u /= np.linalg.norm(u)
        # Пробные шаги
        f_minus = f(*(x_k - g * u))
        f_plus = f(*(x_k + g * u))
        # Основной шаг
        x_new = x_k + step * u * np.sign(f_minus - f_plus)
        # Фиксируем границы
        x_new = np.clip(x_new, [x1_start, x2_start], [x1_stop, x2_stop])
        trajectory.append(x_new.copy())
        if np.linalg.norm(x_new - x_k) < eps:
            break
        x_k = x_new
    return x_k, f(*x_k), np.array(trajectory)


optimum_points, optimum_val, trajectory = gradient_descent_pairwise(
    TargetFunc, initial_point, eps, g, step
)
print(
    f"R(x1, x2) <=> R({optimum_points[0]:.4f}, {optimum_points[1]:.4f}) = {optimum_val:.4f}"
)
x1_vals = np.linspace(x1_start, x1_stop, 100)
x2_vals = np.linspace(x2_start, x2_stop, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = TargetFunc(X1, X2)
plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, Z, levels=35, cmap="viridis")
plt.colorbar()
plt.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    color="red",
    linestyle="dashed",
    marker="o",
    markersize=3,
    label="Траектория",
)
plt.scatter(
    optimum_points[0],
    optimum_points[1],
    color="orange",
    marker="o",
    s=100,
    label=f"Оптимум: ({optimum_points[0]:.4f}, {optimum_points[1]:.4f})",
)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("R(x1, x2)")
plt.legend()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none")
ax.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    TargetFunc(trajectory[:, 0], trajectory[:, 1]),
    color="red",
    linestyle="dashed",
    marker="o",
    markersize=3,
    label="Траектория",
)
ax.scatter(
    optimum_points[0],
    optimum_points[1],
    optimum_val,
    color="orange",
    marker="o",
    s=100,
    label=f"Оптимум: ({optimum_points[0]:.4f}, {optimum_points[1]:.4f})",
)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("R(x1, x2)")
ax.set_title("R(x1, x2)")
ax.legend()
plt.show()
