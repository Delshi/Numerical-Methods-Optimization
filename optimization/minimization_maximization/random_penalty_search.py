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
alpha = 1 # Шаг




# Целевая функция
def TargetFunc(x1, x2):
    return A * x1**3 + B * x2**3 - C * x1 - D * x2
    # return x1**2 - 3*x1*x2 + 10*x2**2 + 5*x1 - 3*x2
    # return (A*x1**2 + B*x2**2 + C*x1*np.sin(D*x1*x2))
    # return A - 2*np.exp(-(x1**2 + x2**2)/(0.1))




def RandomPenaltySearch(f, initial_point, alpha, eps, marks, max_iter):
    '''
    Случайные направления. 
    Начальная точка x_k -> f = R(x_k)
    x_k+1 = x_k + alpha * h
    h -- случайный вектор
    x_k = x_k+1 if R(x_k+1) < R(x_k) else новый вектор и повторение цикла
    Наказание: через расчет вероятностей с параметром регулировки наказания g
    Конец, когда alpha < eps
    '''

    previous_point = initial_point.copy()
    current_point = initial_point.copy()
    current_value = f(*current_point)

    trajectory = [current_point.copy()]

    no_improve = 0
    max_no_improve = 10

    g = 0.3

    for _ in range(max_iter):
        
        r = alpha * np.sqrt(np.random.uniform(0, 1, marks))
        # r /= np.linalg.norm(r)
        theta = np.random.uniform(0, 2*np.pi, marks)
        
        x1_coords = initial_point[0] + r * np.cos(theta)
        x2_coords = initial_point[1] + r * np.sin(theta)
        x1_coords = np.clip(x1_coords, x1_start, x1_stop)
        x2_coords = np.clip(x2_coords, x2_start, x2_stop)

        points = np.vstack((x1_coords, x2_coords)).T

        vals = [f(x1, x2) for x1, x2 in points]

        chance = [np.exp(-g*(val - np.min(vals))) for val in vals]
        chance /= np.sum(chance)

        '''
        numpy choice строит кумулятивную функцию распределения (CDF) для переданных вероятностей,
        генерирует случайное r из [0, 1)
        элемент выбирается путем поиска первого индекса в распределении, где r <= CDF
        Пример: [A, B, C] = [0.1, 0.6, 0.3] => CDF = [0.1, 0.7, 1]; 
        r = 0.5 -> 0.5<=0.1? Нет. 0.5<=0.7? Да. => return B'
        '''
        current_point_idx = np.random.choice(len(points), p=chance)
        current_point = points[current_point_idx]
        current_point = np.clip(current_point, [x1_start, x2_start], [x1_stop, x2_stop])

        if f(*current_point) < f(*previous_point):
            previous_point = current_point
            current_value = f(*current_point)

            trajectory.append(current_point.copy())
        else:
            no_improve += 1
        

        checkout_to_correct_penalty = (vals < f(*current_point)).astype(int)

        if np.mean(checkout_to_correct_penalty) < g:
            g = (g/100)*95
        else:
            g = (g/100)*105

        g = np.clip(g, 0.1, 1)

        if no_improve >= max_no_improve:
            alpha = (alpha/100)*90
            no_improve = 0

        if alpha < eps:
            break

        # print(checkout_to_correct_penalty)
        # print('POINTS\n', points)
        # print('VALS', vals)
        # print('CHANCE', chance)
        # print('SELECTED', current_point)
        # print('PREV', previous_point)
        # print('PARAMETER', g)
        # print('STEP', alpha, '\n')
        # print('ITER', _)

    return previous_point, current_value, np.array(trajectory)




x1_vals = np.linspace(x1_start, x1_stop, 100)
x2_vals = np.linspace(x2_start, x2_stop, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = TargetFunc(X1, X2)


optimum_points, optimum_val, trajectory = RandomPenaltySearch(
    TargetFunc, initial_point, alpha=alpha, eps=eps, marks=50, max_iter=100
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
