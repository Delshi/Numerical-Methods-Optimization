import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
Kx = 15.7  # Коэффициент турбулентной диффузии
dx = 10  # Шаг сетки
dt = 1  # Временной шаг
t_final = 5  # Максимальное время
num_time_steps = int(t_final / dt)  # Всего шагов

# Создание сетки
x_nodes = np.array([0, 10, 20, 30, 40])
num_nodes = len(x_nodes)

# Инициализация концентрации
C = np.zeros(num_nodes)
C[0] = 1.0 / dx  # Начальная концентрация

# Численное решение, явная схема
for step in range(num_time_steps):
    C_new = np.zeros_like(C)
    for i in range(num_nodes):
        if i == 0:
            C_new[i] = C[i] + (Kx * dt / dx**2) * 2 * (C[i + 1] - C[i])
        elif i == num_nodes - 1:
            C_new[i] = 0.0
        else:
            C_new[i] = C[i] + (Kx * dt / dx**2) * (C[i + 1] - 2 * C[i] + C[i - 1])
    C = C_new.copy()

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(x_nodes, C, "bo-", linewidth=2, markersize=8, label="Распределение")
plt.title("Распределение концентрации через 5 секунд", fontsize=14)
plt.xlabel("Расстояние, м", fontsize=12)
plt.ylabel("Концентрация, кг/м", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()
