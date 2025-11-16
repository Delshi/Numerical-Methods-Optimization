import numpy as np
import matplotlib.pyplot as plt

A, B, C, D = 1, 2, 1.2, 2
x_1_start, x_1_stop = -2, 2
x_2_start, x_2_stop = -2, 2
initial_point = np.array([-1.8, -1.5])
step = 0.1
eps = 0.005

def TargetFunc(x1, x2):
    return A*x1**2 + B*x2**2+ C*x1*np.sin(D*x1*x2)

def rosenbrock_method(func, initial_point, step, eps):
    x = np.array(initial_point, dtype=np.float64)
    n = len(x)
    directions = np.eye(n)  # Начальные направления (ортонормированные)

    trajectory = [x.copy()]
    
    while True:
        improved = False
        new_x = x.copy()
        
        # Поиск по каждому направлению
        for i in range(n):
            di = directions[i]
            # Шаг вперед
            candidate = new_x + step * di

            if func(candidate[0], candidate[1]) < func(new_x[0], new_x[1]):
                new_x = candidate
                improved = True
                trajectory.append(candidate.copy())

            else:
                # Шаг назад
                candidate_back = new_x - step * di

                if func(candidate_back[0], candidate_back[1]) < func(new_x[0], new_x[1]):
                    new_x = candidate_back
                    improved = True
                    trajectory.append(candidate_back.copy())
                    

        if not improved:
            # Пересчет направлений
            delta = new_x - x
            delta_norm = np.linalg.norm(delta)

            if delta_norm < eps:
                break

            # Ортогонализация Грама-Шмидта для новых направлений
            v1 = delta / delta_norm
            v2 = np.array([-v1[1], v1[0]])  # Перпендикулярный вектор
            v2_norm = np.linalg.norm(v2)

            directions = np.array([v1, v2])

            print('TURN')
            
        else:
            x = new_x.copy()
 
    
    return x, np.array(trajectory)  # Преобразуем trajectory в массив NumPy

optimum_points, trajectory = rosenbrock_method(TargetFunc, initial_point, step, eps)
optimum_val = TargetFunc(optimum_points[0], optimum_points[1])

print(f"R(x1, x2) <=> R({optimum_points[0]:.20f}, {optimum_points[1]:.20f}) = {optimum_val:.48f}")

x1_vals = np.linspace(x_1_start, x_1_stop, 100)
x2_vals = np.linspace(x_2_start, x_2_stop, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = TargetFunc(X1, X2)

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
