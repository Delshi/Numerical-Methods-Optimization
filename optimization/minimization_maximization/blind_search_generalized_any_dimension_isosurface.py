import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from scipy.spatial import Delaunay

# Параметры задачи (добавлены параметры для третьей переменной)
A, B, C, D, E, F = 2, 3, 4, 4, 5, 5
initial_point = np.array([-1.0, 1.0, 0.5])  # 3D начальная точка
x_bounds = [(-2, 2), (-2, 2), (-2, 2)]  # Границы для всех переменных
eps = 0.01
alpha = 5


# Обобщённая целевая функция для трёх переменных
def TargetFunc(*x):
    return (
        A * x[0] ** 3 + B * x[1] ** 3 + E * x[2] ** 3 - C * x[0] - D * x[1] - F * x[2]
    )


def BlindSearch(f, initial_point, alpha, eps, marks, max_iter):
    """
    Обобщённая версия алгоритма для N-мерного пространства
    """
    current_point = initial_point.copy()
    n_dim = len(current_point)
    global_bounds = np.array(x_bounds)

    trajectory = [current_point.copy()]
    values = [f(*current_point)]

    for _ in range(max_iter):
        # Генерация случайных точек в окрестности
        lb = current_point - alpha
        ub = current_point + alpha

        # Генерируем точки с учётом глобальных границ
        coords = []
        for i in range(n_dim):
            points = np.random.uniform(lb[i], ub[i], marks)
            coords.append(np.clip(points, *global_bounds[i]))

        test_points = np.column_stack(coords)

        # Вычисление значений функции
        values = np.array([f(*point) for point in test_points])
        min_idx = np.argmin(values)
        min_point = test_points[min_idx]
        min_value = values[min_idx]

        # Обновление текущей точки
        if min_value < f(*current_point):
            current_point = min_point
            trajectory.append(current_point.copy())

        # Проверка условия останова
        if len(trajectory) > 1:
            delta = np.linalg.norm(trajectory[-1] - trajectory[-2])
            if delta < eps:
                break

    return current_point, f(*current_point), np.array(trajectory)


# Визуализация только для 2D случаев
if len(initial_point) == 2:
    x1_vals = np.linspace(*x_bounds[0], 100)
    x2_vals = np.linspace(*x_bounds[1], 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = TargetFunc(X1, X2)

    # Поиск
    optimum_points, optimum_val, trajectory = BlindSearch(
        TargetFunc, initial_point, alpha=alpha, eps=eps, marks=500, max_iter=500
    )

    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, Z, levels=35, cmap="viridis")
    plt.colorbar()
    plt.plot(trajectory[:, 0], trajectory[:, 1], "r.-", label="Траектория")
    plt.scatter(
        *optimum_points[:2],
        c="red",
        s=100,
        label=f"Оптимум: ({optimum_points[0]:.2f}, {optimum_points[1]:.2f})",
    )
    plt.legend()
    plt.show()

elif len(initial_point) == 3:
    # Поиск
    optimum_points, optimum_val, trajectory = BlindSearch(
        TargetFunc, initial_point, alpha=alpha, eps=eps, marks=100, max_iter=100
    )

    print(f"Найденный оптимум: {optimum_points}")
    print(f"Значение функции: {optimum_val:.4f}")

    # Параметры визуализации
    grid_size = 30
    n_levels = 10  # 4 уровня выше и ниже + оптимальный
    base_alpha = 0.5
    # cmap = plt.cm.coolwarm
    cmap = plt.cm.viridis
    arrow_style = {"color": "cyan", "linewidth": 2}

    x = np.linspace(*x_bounds[0], grid_size)
    y = np.linspace(*x_bounds[1], grid_size)
    z = np.linspace(*x_bounds[2], grid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    V = TargetFunc(X, Y, Z)

    # Автоматическое определение уровней
    opt_val = TargetFunc(*optimum_points)
    v_min, v_max = V.min(), V.max()
    levels = np.linspace(v_min, v_max, n_levels)
    level_colors = cmap(np.linspace(0, 1, n_levels))

    # Фигура и оси
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Построение изоповерхностей
    for i, (level, color) in enumerate(zip(levels, level_colors)):
        try:
            # Вычисление изоповерхности
            verts, faces, _, _ = measure.marching_cubes(
                V,
                level=level,
                spacing=(
                    np.ptp(x) / grid_size,
                    np.ptp(y) / grid_size,
                    np.ptp(z) / grid_size,
                ),
            )

            # Корректировка координат
            verts += [x.min(), y.min(), z.min()]

            # Визуализация
            alpha = base_alpha * (
                1 + 2 * (0.5 - abs(level - opt_val) / (v_max - v_min))
            )
            ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                faces,
                verts[:, 2],
                color=color,
                alpha=alpha,
                edgecolor="none",
                antialiased=True,
            )

        except RuntimeError:
            continue

    # Траектория поиска
    for i in range(1, len(trajectory)):
        start = trajectory[i - 1]
        end = trajectory[i]
        ax.quiver(
            start[0],
            start[1],
            start[2],
            end[0] - start[0],
            end[1] - start[1],
            end[2] - start[2],
            color="lime",
            arrow_length_ratio=0.1,
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
        )

    # Точка оптимума
    ax.scatter(
        *optimum_points,
        s=250,
        c="gold",
        marker="*",
        edgecolor="black",
        depthshade=False,
        label=f"Optimum: ({optimum_points[0]:.3f}, "
        f"{optimum_points[1]:.3f}, "
        f"{optimum_points[2]:.3f})",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(v_min, v_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, aspect=40)
    cbar.set_label("Function Value", rotation=270, labelpad=20)

    ax.view_init(elev=30, azim=-60)
    ax.dist = 10
    ax.xaxis.pane.set_alpha(0.8)
    ax.yaxis.pane.set_alpha(0.8)
    ax.zaxis.pane.set_alpha(0.8)
    ax.xaxis.pane.set_color("white")
    ax.yaxis.pane.set_color("white")
    ax.zaxis.pane.set_color("white")
    ax.set_box_aspect([1, 1, 1])

    ax.set_title(
        f"3D Изоповерхности\n({n_levels} уровней, α={alpha}, ε={eps})",
        fontsize=14,
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 0.9))

    plt.tight_layout()
    plt.show()
