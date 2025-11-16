import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# Параметры задачи
A, B, C, D = 2, 3, 4, 4
initial_point = np.array([-1, 1])
x1_start, x1_stop = -2, 2
x2_start, x2_stop = -2, 2
eps = 0.01  # Погрешность
g = 0.01    # Параметр шага для парной пробы
alpha = 0.1 # Начальный шаг




# Целевая функция
def TargetFunc(x1, x2):
    return A * x1**3 + B * x2**3 - C * x1 - D * x2
    # return A - 2*np.exp(-(x1**2 + x2**2)/(0.1))




# def PairwiseProbeGrad(f, x, g):
#     '''
#     Парная проба. 
#     Вместо случайных векторов было решено взять базисные единичные, 
#     чтобы не вредить стабильности метода сопряженных градиентов дополнительной стохаистичностью
#     '''
#     grad = np.zeros_like(x)

#     for i in range(len(x)):
#         e_i = np.zeros_like(x)
#         e_i[i] = g
#         grad[i] = (f(*(x + e_i)) - f(*(x - e_i))) / (2 * g)

#     return grad


# def PairwiseProbeGrad(f, x, g):
#     '''
#     Классическая парная проба
#     '''
#     grad = np.zeros_like(x)
    
#     e_i = np.random.uniform(-1, 1, x.shape)
#     e_i /= np.linalg.norm(e_i)

#     delta = (f(*(x + g*e_i)) - f(*(x - g*e_i))) / (2 * g)

#     grad[:] = delta * e_i

#     grad /= np.linalg.norm(grad)

#     return grad


def PairwiseProbeGrad(f, x, g, num_samples=50):
    """
    Парная проба.
    Я немного улучшил классический алгоритм: вместо выбора одного из двух случайных направлений,
    используется num_samples случайных направлений с последующим выбором лучшего.
    Стабильность несколько ниже, чем при использовании базисных единичных векторов.
    """
    grad = np.zeros_like(x)
    probes = {}
    
    for _ in range(num_samples):
        direction = np.random.uniform(-1, 1, *x.shape)
        direction /= np.linalg.norm(direction)

        f_plus = f(*(x + g * direction))
        f_minus = f(*(x - g * direction))
        
        grad += ((f_plus - f_minus) / (2 * g)) * direction
        probes[TargetFunc(*grad)] = grad

    direction = probes.get(min(probes))

    return direction




def ConjugateGrads(f, initial_point, eps, g, alpha, max_iter=500):
    x_k = np.array(initial_point, dtype=np.float64)

    grad_k = PairwiseProbeGrad(f, x_k, g) # Градиент в точке
    d_k = -grad_k  # Направление поиска

    c1, c2 = 0.1, 0.9
    max_step_size = 0.5  # Ограничение максимального шага

    n_rest = 2 * len(initial_point) + 1

    trajectory = [x_k.copy()]
    

    for _ in range(max_iter):

        if np.linalg.norm(grad_k) < eps:
            break
        

        # Линейный поиск по Вольфе
        alpha_k = alpha
        '''
        while:
            func(x_old + alpha * direction) > [ func(x_old) + c1 * alpha * <old_grad, direction> ]
            and
            |<new_grad, new_grad>| > c2 * |<new_grad, old_grad>|
        do:
            alpha = alpha * 0.5
        '''
        while ( 
                (f(*(x_k + alpha_k * d_k)) > f(*x_k) + c1 * alpha_k * np.dot(grad_k, d_k))
                and
                (np.abs(np.dot(PairwiseProbeGrad(f, x_k + alpha_k * d_k, g), d_k)) > c2 * np.abs(np.dot(grad_k, d_k)))
                ):

            alpha_k *= 0.5       
            

        # Ограничиваем максимальный шаг
        step = alpha_k * d_k # Шаг step -- вектор смещения
        if np.linalg.norm(step) > max_step_size:
            step *= max_step_size / np.linalg.norm(step)


        x_new = x_k + step
        x_new = np.clip(x_new, [x1_start, x2_start], [x1_stop, x2_stop])
        grad_new = PairwiseProbeGrad(f, x_new, g)

        '''
        beta_k = <grad_new, grad_new - grad_old> / <grad_old, grad_old>
        '''
        beta_k = np.dot(grad_new, grad_new - grad_k) / np.dot(grad_k, grad_k)


        # Обновление направления каждые 2n шагов 
        if _ % n_rest == 0: 
            beta_k = 0

        d_new = -grad_new + beta_k * d_k
        

        # Проверка на корректность направления
        if np.dot(d_new, grad_new) > 0:
            d_new = -grad_new
        

        trajectory.append(x_new.copy())
        

        # Остановка по погрешности
        if np.linalg.norm(x_new - x_k) < eps: 
            break
        

        x_k, grad_k, d_k = x_new, grad_new, d_new


    return x_k, f(*x_k), np.array(trajectory)




optimum_points, optimum_val, trajectory = ConjugateGrads(TargetFunc, initial_point, eps, g, alpha)
print(f"R(x1, x2) <=> R({optimum_points[0]:.4f}, {optimum_points[1]:.4f}) = {optimum_val:.4f}")


x1_vals = np.linspace(x1_start, x1_stop, 100)
x2_vals = np.linspace(x2_start, x2_stop, 100)
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
ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.8)
ax.plot(trajectory[:, 0], trajectory[:, 1], TargetFunc(trajectory[:, 0], trajectory[:, 1]), color='red', linestyle='dashed', marker='o', markersize=3, label='Траектория')
ax.scatter(optimum_points[0], optimum_points[1], optimum_val, color='orange', marker='o', s=100, label=f'Оптимум: ({optimum_points[0]:.4f}, {optimum_points[1]:.4f})')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('R(x1, x2)')
ax.set_title('R(x1, x2)')
ax.legend()
plt.show()
