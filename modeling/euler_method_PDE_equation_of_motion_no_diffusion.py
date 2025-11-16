import matplotlib.pyplot as plt

# Параметры задачи
x_points = [0, 10, 20]  # узлы
sources = [0.74, 0.83, 0.52]  # коэффициенты источников
k_coeff = [0.09, 0.08, 0.08]  # коэффициенты вывода

# Инициализация концентраций
C = [0.0, 0.0, 0.0]

# Временные параметры
dt = 1  # величина временного шага
total_time = 5  # общее время
steps = total_time // dt  # всего шагов

# Численное решение методом Эйлера
for _ in range(steps):
    for i in range(3):
        dCdt = sources[i] - k_coeff[i] * C[i]
        C[i] += dt * dCdt

# Построение графика
plt.figure(figsize=(8, 5))
plt.plot(x_points, C, "o-", markersize=8, label="Концентрация через 5 с")
plt.xlabel("Расстояние, м", fontsize=12)
plt.ylabel("Концентрация", fontsize=12)
plt.title("Распределение концентрации активной примеси через 5 секунд", fontsize=14)
plt.xticks(x_points)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()
