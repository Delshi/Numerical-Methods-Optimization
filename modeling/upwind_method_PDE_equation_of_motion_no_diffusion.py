import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
Vx = 6.3  # Скорость ветра
dx = 10  # Шаг сетки
dt = 1  # Шаг по времени
total_time = 5  # Общее время
num_steps = int(total_time / dt)  # Количество шагов
num_nodes = 6  # Количество узлов сетки (0, 10, 20, 30, 40, 50)

# Число Куранта
Cr = Vx * dt / dx

# Инициализация массива концентраций
C = np.zeros(num_nodes)
C[0] = 1.0 / dx  # Начальная концентрация в первом узле

# Массивы для хранения истории массы
mass_history = []

# Численное решение методом upwind
for _ in range(num_steps):
    C_new = np.zeros_like(C)
    for i in range(num_nodes):
        if i == 0:
            C_new[i] = C[i] * (1 - Cr)
        else:
            C_new[i] = C[i] * (1 - Cr) + Cr * C[i - 1]
    C = C_new.copy()
    mass_history.append(np.sum(C) * dx)  # Сохраняем массу для графика

# Создание графиков
plt.figure(figsize=(12, 5))

# График распределения концентрации
plt.subplot(1, 2, 1)
x_points = np.arange(0, 60, 10)
plt.bar(x_points, C, width=dx, align="edge", edgecolor="k", alpha=0.7)
plt.title("Распределение концентрации через 5 секунд")
plt.xlabel("Расстояние (м)")
plt.ylabel("Концентрация (кг/м)")
plt.grid(linestyle="--", alpha=0.7)
plt.tight_layout()

# График изменения массы
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, num_steps + 1), mass_history, "o-", color="orange")
plt.title("Сохранение массы")
plt.xlabel("Время (с)")
plt.ylabel("Масса (кг)")
plt.ylim(0.9, 1.1)
plt.grid(linestyle="--", alpha=0.7)
plt.axhline(1.0, color="red", linestyle="--", linewidth=1)
plt.show()
