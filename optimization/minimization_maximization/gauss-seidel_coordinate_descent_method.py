import numpy as np
import matplotlib.pyplot as plt


A, B, C, D = 1, 2, 1.2, 2
x_1_start, x_1_stop = -2, 2
x_2_start, x_2_stop = -2, 2
initial_point = np.array([-1.8, -1.2])
step = 0.1
eps = 0.005


def TargetFunc(x1, x2):
    return A * x1**2 + B * x2**2 + C * x1 * np.sin(D * x1 * x2)
    # return A - 2*np.exp(-(x1**2 + x2**2)/(0.1))


def CoordDesc(func, initial_point, step, eps):
    current_point = initial_point.copy()
    trajectory = [current_point.copy()]

    while True:
        prev_point = current_point.copy()

        # Оптимизация по x1
        x1, x2 = current_point
        f_current = func(x1, x2)
        f_plus = func(x1 + step, x2)
        f_minus = func(x1 - step, x2)

        if f_plus < f_current and f_plus <= f_minus:
            while True:
                new_x1 = x1 + step
                new_f = func(new_x1, x2)

                if new_f < f_current:
                    x1, f_current = new_x1, new_f
                else:
                    break

        elif f_minus < f_current:
            while True:
                new_x1 = x1 - step
                new_f = func(new_x1, x2)

                if new_f < f_current:
                    x1, f_current = new_x1, new_f
                else:
                    break

        current_point[0] = x1

        # Оптимизация по x2
        x1, x2 = current_point
        f_current = func(x1, x2)
        f_plus = func(x1, x2 + step)
        f_minus = func(x1, x2 - step)

        if f_plus < f_current and f_plus <= f_minus:
            while True:
                new_x2 = x2 + step
                new_f = func(x1, new_x2)

                if new_f < f_current:
                    x2, f_current = new_x2, new_f
                else:
                    break

        elif f_minus < f_current:
            while True:
                new_x2 = x2 - step
                new_f = func(x1, new_x2)

                if new_f < f_current:
                    x2, f_current = new_x2, new_f
                else:
                    break

        current_point[1] = x2

        trajectory.append(current_point.copy())

        if np.linalg.norm(current_point - prev_point) < eps:
            break

    return current_point, func(*current_point), np.array(trajectory)


x1_vals = np.linspace(x_1_start, x_1_stop, 100)
x2_vals = np.linspace(x_2_start, x_2_stop, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = TargetFunc(X1, X2)


optimum_points, optimum_val, trajectory = CoordDesc(
    TargetFunc, initial_point, step, eps
)
print(
    f"R(x1, x2) <=> R({optimum_points[0]:.32f}, {optimum_points[1]:.32f}) = {optimum_val:.32f}"
)


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
    label=f"Оптимум: ({optimum_points[0]:.32f}, {optimum_points[1]:.32f})",
)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("R(x1, x2)")
plt.legend()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none", alpha=0.9)
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
