import numpy as np
import matplotlib.pyplot as plt


# Параметры задачи
A, B, C, D = 2, 3, 4, 4
initial_point = np.array([-1.0, 1.0])
# initial_point = np.array([9.2, 1.4])
x1_start, x1_stop = -2, 2
x2_start, x2_stop = -2, 2
# x1_start, x1_stop = -10, 10
# x2_start, x2_stop = -10, 10
eps = 0.01 # Погрешность
alpha = 0.1 # Шаг (размер окрестности)
# alpha = 0.9 # Шаг (размер окрестности)




# Целевая функция
def TargetFunc(x1, x2):
    return A * x1**3 + B * x2**3 - C * x1 - D * x2
    # return x1**2 - 3*x1*x2 + 10*x2**2 + 5*x1 - 3*x2
    # return (A*x1**2 + B*x2**2 + C*x1*np.sin(D*x1*x2))
    # return A - 2*np.exp(-(x1**2 + x2**2)/(0.1))


def BlindSearch(f, initial_point, alpha, eps, marks, max_iter):
    '''
        Слепой поиск. Окрестность выбрал квадратную.
        В окрестности генерируется marks случайных точек (распределение выбрал равномерное).
        В каждой точке считается значение функции.
        Выбирается лучшая точка путем сравнения f(new_point) < f(old_point)
        Все повторяется.
        Конец, когда норма разности текущей и предыдущей точки меньше eps.
    '''
    current_point = initial_point.copy()
    previous_point = initial_point.copy()
    current_value = f(*current_point)

    trajectory = [current_point.copy()]


    for _ in range(max_iter):
        # Делаем квадратную окрестность
        lb_x1 = current_point[0] - alpha
        ub_x1 = current_point[0] + alpha
        lb_x2 = current_point[1] - alpha
        ub_x2 = current_point[1] + alpha


        # Случайные точки в окрестности
        x1_coords = np.random.uniform(lb_x1, ub_x1, marks)
        x2_coords = np.random.uniform(lb_x2, ub_x2, marks)

        x1_coords = np.clip(x1_coords, x1_start, x1_stop)
        x2_coords = np.clip(x2_coords, x2_start, x2_stop)

        # Собираем компоненты попарно
        test_points = np.column_stack((x1_coords, x2_coords))
        

        # Считаем значения функции в точках окрестности
        values = np.array([f(x1, x2) for x1, x2 in test_points])
        min_idx = np.argmin(values)
        min_value = values[min_idx]
        min_point = test_points[min_idx]
        

        # Обновляем текущую точку, если найдена лучше
        if min_value < current_value:
            current_point = min_point
            current_value = min_value

            trajectory.append(current_point.copy())


        if np.linalg.norm(current_point - previous_point) < eps:
            break
        
        previous_point = current_point.copy()


    return current_point, current_value, np.array(trajectory)


x1_vals = np.linspace(x1_start, x1_stop, 100)
x2_vals = np.linspace(x2_start, x2_stop, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = TargetFunc(X1, X2)


optimum_points, optimum_val, trajectory = BlindSearch(
    TargetFunc, initial_point, alpha=alpha, eps=eps, marks=100, max_iter=100
)
print(f"R(x1, x2) <=> R({optimum_points[0]:.6f}, {optimum_points[1]:.6f}) = {optimum_val:.6f}")


plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, Z, levels=35, cmap='viridis')
plt.colorbar()
plt.plot(trajectory[:, 0], trajectory[:, 1], color='red', linestyle='dashed', marker='o', markersize=3, label='Траектория')
plt.scatter(optimum_points[0], optimum_points[1], color='orange', marker='o', s=100, label=f'Оптимум: ({optimum_points[0]:.4f}, {optimum_points[1]:.4f})')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('R(x1, x2)')
plt.legend()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.9)
ax.plot(trajectory[:, 0], trajectory[:, 1], TargetFunc(trajectory[:, 0], trajectory[:, 1]), color='red', linestyle='dashed', marker='o', markersize=3, label='Траектория')
ax.scatter(optimum_points[0], optimum_points[1], optimum_val, color='orange', marker='o', s=100, label=f'Оптимум: ({optimum_points[0]:.4f}, {optimum_points[1]:.4f})')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('R(x1, x2)')
ax.set_title('R(x1, x2)')
ax.legend()
plt.show()
